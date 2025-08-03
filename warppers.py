from typing import Callable, Any, Optional, Dict
import os
import time
import inspect
import warnings
import functools
import threading
import torch
import torch.distributed as dist
import gc

ENV_VAR_NAME = 'DEBUG_CALLS_N'
DEFAULT_N_VALUE = 0

def debug_calls(only_rank0=True):
    """
    A decorator to control the frequency of debug calls based on an environment variable.
    The decorator checks the environment variable 'DEBUG_CALLS_N' to determine how often
    to call the decorated function. If the variable is set to a positive integer, the function
    will be called every n times. If the variable is not set or set to a non-positive integer,
    the function will not be called at all.

    The decorator behavior regarding distributed settings is controlled by the `only_rank0` parameter:
    - If `only_rank0` is True (default): The function is only potentially called from the main
      process (rank 0). Counts are maintained only on rank 0.
    - If `only_rank0` is False: The function is potentially called on all ranks. Each rank
      maintains its own independent call count.

    Args:
        only_rank0 (bool): If True, only rank 0 will count and execute the decorated function.
                           If False, all ranks will count and execute independently.
                           Defaults to True.

    Usage:
    ```
    # Example 1: Only debug on rank 0, every N calls
    @debug_calls() # or @debug_calls(only_rank0=True)
    def my_function_rank0_only_debug():
        print(f"my_function_rank0_only_debug called")
        pass

    # Example 2: Debug on all ranks, every N calls (each rank counts independently)
    @debug_calls(only_rank0=False)
    def my_function_all_ranks_debug(call_count=None):
        # Note: LOCAL_RANK needs to be fetched inside the function if needed for printing
        local_rank_for_print = os.getenv('LOCAL_RANK', '0')
        print(f"[Rank {local_rank_for_print}] my_function_all_ranks_debug called, decorator count: {call_count}")
        pass
    ```
    """

    n_str = os.getenv(ENV_VAR_NAME)
    n = DEFAULT_N_VALUE  # Default to not calling if var is not set

    if n_str is not None:  # Env var is set
        try:
            parsed_n = int(n_str)
            # We only care about positive n for frequency.
            # If n_str is "0" or "-1", n will remain DEFAULT_N_VALUE (e.g., 0)
            # or become the parsed non-positive value, leading to no calls.
            n = parsed_n
        except ValueError:
            warnings.warn(
                f"Invalid value '{n_str}' for env var {ENV_VAR_NAME}. "
                f"Using n={DEFAULT_N_VALUE} (debug calls likely disabled)."
            )
            n = DEFAULT_N_VALUE  # Fallback to default on error

    def decorator(func):
        # Each decorated function instance, per process, will get its own counter.
        # This is important for the case when only_rank0=False.
        counter = {'count': 0}
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_rank_str = os.getenv('LOCAL_RANK', '0')
            try:
                local_rank = int(local_rank_str)
            except ValueError:
                warnings.warn(
                    f"Invalid LOCAL_RANK value '{local_rank_str}'. Assuming rank 0. "
                    "This might lead to unexpected behavior in distributed settings if not actually rank 0."
                )
                local_rank = 0 # Fallback, but with a warning

            # Condition to skip execution entirely
            if n <= 0:  # If frequency is non-positive, never call
                return None

            if only_rank0 and local_rank != 0:
                # If only_rank0 is true, and we are not on the main process, skip.
                return None

            # If we reach here, this rank is eligible to count and potentially execute.
            # For only_rank0=True, this block is only reached by rank 0.
            # For only_rank0=False, this block is reached by all ranks.
            counter['count'] += 1

            if counter['count'] % n == 0:
                effective_call_count = counter['count']
                # if func needs call_count, pass it
                if 'call_count' in param_names:
                    kwargs['call_count'] = effective_call_count
                
                # Optionally, print from the decorator for its own debugging
                # prefix = f"[Rank {local_rank}] " if not only_rank0 else "[Rank 0] "
                # print(f"{prefix}Decorator: Calling {func.__name__} (count: {effective_call_count})")

                return func(*args, **kwargs)
            return None # Not the Nth call for this rank
        return wrapper
    return decorator



# Global thread-local storage for logger activation states
_thread_local_storage = threading.local()

def _get_active_loggers_dict() -> Dict[str, bool]:
    """Get the dictionary of active loggers for the current thread."""
    if not hasattr(_thread_local_storage, 'active_loggers'):
        _thread_local_storage.active_loggers = {}
    return _thread_local_storage.active_loggers

def _is_logger_globally_active(logger_name: str) -> bool:
    """
    Check if the logger with the specified name is active in the current thread.
    Defaults to False (disabled) if not explicitly set.
    """
    return _get_active_loggers_dict().get(logger_name, False)

def _set_logger_globally_active(logger_name: str, active: bool):
    """Set the activation state of the specified logger for the current thread."""
    _get_active_loggers_dict()[logger_name] = active


class LoggerControl:
    """
    A generic context manager for controlling the activation state of any logger.
    """
    def __init__(self, logger_name: str, activate: bool):
        self.logger_name = logger_name
        self.activate_in_context = activate
        self.previous_state: Optional[bool] = None

    def __enter__(self):
        self.previous_state = _is_logger_globally_active(self.logger_name)
        _set_logger_globally_active(self.logger_name, self.activate_in_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_state is not None:
            _set_logger_globally_active(self.logger_name, self.previous_state)
        return False



def time_logger_enabled():
    return LoggerControl('time', True)

def time_logger_disabled():
    return LoggerControl('time', False)

def time_logger_set_active(active: bool):
    return LoggerControl('time', active)



def _find_device_from_args(*args, **kwargs) -> Optional[torch.device]:
    """
    Find the first CUDA tensor in the function arguments and return its device.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            return arg.device
    for kwarg in kwargs.values():
        if isinstance(kwarg, torch.Tensor) and kwarg.is_cuda:
            return kwarg.device
    # If not found, check if there is a default CUDA device as a fallback
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return None


REGISTERED_TIME_LOGGERS: Dict[str, Callable] = {}


def time_logger(func: Callable) -> Callable:
    """
    A decorator that logs the time taken by a function to execute
    and keeps track of the average time taken across multiple calls,
    if the logger is active (controlled by TimeLoggerControl context manager).
    By default, the logger is INACTIVE.

    MODIFIED: This decorator now automatically registers the decorated function
    into a global registry, allowing for runtime inspection of all timed functions.

    The average time, call count, and current duration can be accessed via the
    `get_average_time`, `get_call_count`, and `get_current_duration`
    methods of the wrapped function. If the logger is disabled for a call,
    these stats are not updated for that call.
    """
    call_count: int = 0
    current_average: float = 0.0
    current_duration: float = 0.0

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        nonlocal call_count, current_average, current_duration

        if _is_logger_globally_active('time'):
            device = _find_device_from_args(*args, **kwargs)
            if device is None or device.type != 'cuda':
                # If no CUDA tensor is found in the arguments and CUDA is not available,
                warnings.warn(
                    f"Time logger for '{func.__qualname__}' skipped: "
                    f"No CUDA tensor found in arguments and CUDA not available."
                )
                return func(*args, **kwargs)
            try:
                start_event = torch.Event(device=device, enable_timing=True)
                end_event = torch.Event(device=device, enable_timing=True)
            except AttributeError:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # end_time = time.perf_counter()
                end_event.record()
                torch.cuda.synchronize(device)
                duration = start_event.elapsed_time(end_event)
                call_count += 1
                current_average += (duration - current_average) / call_count
                current_duration = duration
        else:
            return func(*args, **kwargs)

    def get_average_time() -> float:
        return current_average

    def get_call_count() -> int:
        return call_count

    def get_current_duration() -> float:
        return current_duration

    def reset_stats():
        """Resets the statistics for this specific timed function."""
        nonlocal call_count, current_average, current_duration
        call_count = 0
        current_average = 0.0
        current_duration = 0.0

    wrapper.get_average_time = get_average_time
    wrapper.get_call_count = get_call_count
    wrapper.get_current_duration = get_current_duration
    wrapper.reset_stats = reset_stats
    wrapper._original_function = func

    registry_key = f"{func.__module__}.{func.__qualname__}"
    if registry_key in REGISTERED_TIME_LOGGERS:
        warnings.warn(
            f"Function '{registry_key}' is already registered with @time_logger. "
            "This will overwrite the previous registration."
        )
    REGISTERED_TIME_LOGGERS[registry_key] = wrapper

    return wrapper


def get_all_time_logger_stats(called_only: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves statistics for all functions decorated with @time_logger.

    Returns:
        A dictionary where keys are the unique names of the functions
        and values are dictionaries containing their stats.
    """
    stats = {}
    for name, logger_func in REGISTERED_TIME_LOGGERS.items():
        if called_only and logger_func.get_call_count() == 0:
            continue
        stats[name] = {
            "call_count": logger_func.get_call_count(),
            "average_time_ms": logger_func.get_average_time(),
            "last_duration_ms": logger_func.get_current_duration(),
        }
    return stats

def reset_all_time_logger_stats():
    """
    Resets the statistics for all functions decorated with @time_logger.
    """
    for logger_func in REGISTERED_TIME_LOGGERS.values():
        logger_func.reset_stats()
        
        
REGISTERED_MEMORY_LOGGERS: Dict[str, Callable] = {}


def memory_logger(func: Callable) -> Callable:
    """
    A decorator that logs the peak memory usage of a function when it is called,
    and keeps track of the average peak memory usage across multiple calls,
    if the logger is active (controlled by MemoryLoggerControl context manager).
    By default, the logger is INACTIVE.
    """
    call_count: int = 0
    current_max_peak_allocated_mem: float = 0.0
    current_max_reserved_mem: float = 0.0


    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        nonlocal call_count, current_max_peak_allocated_mem, current_max_reserved_mem

        if _is_logger_globally_active('memory'):
            device = _find_device_from_args(*args, **kwargs)
            if device is None or device.type != 'cuda':
                # If no CUDA tensor is found in the arguments and CUDA is not available,
                warnings.warn(
                    f"Memory logger for '{func.__qualname__}' skipped: "
                    f"No CUDA tensor found in arguments and CUDA not available."
                )
                return func(*args, **kwargs)

            # 1. Reset the peak memory stats for the device
            torch.cuda.reset_peak_memory_stats(device)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                torch.cuda.synchronize(device)  # Ensure all operations are completed
                # 2. Get the peak memory usage (bytes) since the reset
                peak_bytes = torch.cuda.max_memory_allocated(device)
                reserved_bytes = torch.cuda.memory_reserved(device)
                
                # 3. Update the statistics
                call_count += 1
                current_max_peak_allocated_mem = max(current_max_peak_allocated_mem, peak_bytes)
                current_max_reserved_mem = max(current_max_reserved_mem, reserved_bytes)

        else:
            return func(*args, **kwargs)

    def get_max_peak_allocated_memory() -> float:
        """
        Returns the maximum peak memory allocated in bytes across all calls.
        """
        return current_max_peak_allocated_mem
    
    def get_max_reserved_memory() -> float:
        """
        Returns the maximum reserved memory in bytes across all calls.
        """
        return current_max_reserved_mem

    def get_call_count() -> int:
        return call_count

    def reset_stats():
        """Resets the statistics for this specific memory logger."""
        nonlocal call_count, current_max_peak_allocated_mem, current_max_reserved_mem
        call_count = 0
        current_max_peak_allocated_mem = 0.0
        current_max_reserved_mem = 0.0
        torch.cuda.empty_cache()
        gc.collect()
        

    wrapper.get_max_peak_allocated_memory = get_max_peak_allocated_memory
    wrapper.get_max_reserved_memory = get_max_reserved_memory
    wrapper.get_call_count = get_call_count
    wrapper.reset_stats = reset_stats
    wrapper._original_function = func

    registry_key = f"{func.__module__}.{func.__qualname__}"
    if registry_key in REGISTERED_MEMORY_LOGGERS:
        warnings.warn(
            f"Function '{registry_key}' is already registered with @memory_logger. "
            "This will overwrite the previous registration."
        )
    REGISTERED_MEMORY_LOGGERS[registry_key] = wrapper

    return wrapper


def memory_logger_enabled():
    """Returns a context manager to enable @memory_logger in this block."""
    return LoggerControl('memory', True)

def memory_logger_disabled():
    """Returns a context manager to disable @memory_logger in this block."""
    return LoggerControl('memory', False)

def memory_logger_set_active(active: bool):
    """Returns a context manager to set the status of @memory_logger."""
    return LoggerControl('memory', active)

def _format_bytes(num_bytes: int) -> str:
    """
    Formats bytes into a human-readable string with appropriate units.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"

def get_all_memory_logger_stats(called_only: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the statistics for all functions decorated with @memory_logger.
    """
    stats = {}
    for name, logger_func in REGISTERED_MEMORY_LOGGERS.items():
        if called_only and logger_func.get_call_count() == 0:
            continue
        stats[name] = {
            "call_count": logger_func.get_call_count(),
            "max_peak_allocated_memory": _format_bytes(logger_func.get_max_peak_allocated_memory()),
            "max_reserved_memory": _format_bytes(logger_func.get_max_reserved_memory()),
        }
    return stats

def reset_all_memory_logger_stats():
    """
    Resets the statistics for all functions decorated with @memory_logger.
    """
    for logger_func in REGISTERED_MEMORY_LOGGERS.values():
        logger_func.reset_stats()




class OOMHandledError(Exception):
    pass


def oom_resilient(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_distributed = dist.is_available() and dist.is_initialized()
        local_rank = os.getenv('LOCAL_RANK', '0')
        
        local_success = True
        return_value = None
        try:
            return_value = func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            local_success = False
            print(f"Local rank {local_rank}: OOM detected!", flush=True)
        except Exception as e:
            local_success = False
            print(f"Local rank {local_rank}: Exception: {e}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            if is_distributed:
                failure_indicator = torch.tensor(
                    [0.0 if local_success else 1.0], 
                    device=f"cuda:{local_rank}"
                )
                
                dist.all_reduce(failure_indicator, op=dist.ReduceOp.SUM)
                if failure_indicator.item() > 0:
                    local_success = False

        if not local_success:
            raise OOMHandledError("OOM was handled. The caller should retry with a new strategy.")
        return return_value
        
    return wrapper


def oom_handler(func):
    resilient_func = oom_resilient(func)

    @functools.wraps(func)
    def final_wrapper(*args, **kwargs):
        local_rank = os.getenv('LOCAL_RANK', '0')
        try:
            return resilient_func(*args, **kwargs)
        except OOMHandledError as e:
            print(f"[oom_handler - Local rank {local_rank}]: Caught OOMHandledError. Task failed gracefully. Returning None.", flush=True)
            return None
    return final_wrapper


@memory_logger
@time_logger
def gpu_intensive_task(tensor_a, tensor_b):
    torch.matmul(tensor_a, tensor_b.T)
    torch.nn.functional.cosine_similarity(tensor_a, tensor_b, dim=1)
    torch.norm(tensor_a - tensor_b, dim=1)
    return None



if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU example.")
    else:
        device = torch.device("cuda")
        N, C = 6400, 2048
        a = torch.randn(N, C, device=device)
        b = torch.randn(N, C, device=device)

        print("--- First run (Loggers disabled) ---")
        result = gpu_intensive_task(a, b)
        print("Time Stats:", get_all_time_logger_stats(called_only=True))
        print("Memory Stats:", get_all_memory_logger_stats(called_only=True))
        print("-" * 20)


        print("\n--- Second run (Memory Logger enabled) ---")
        with memory_logger_enabled():
            result = gpu_intensive_task(a, b)
        print("Time Stats:", get_all_time_logger_stats(called_only=True))
        print("Memory Stats:", get_all_memory_logger_stats(called_only=True))
        print("-" * 20)


        print("\n--- Third run (All Loggers enabled) ---")
        with time_logger_enabled(), memory_logger_enabled():
            for _ in range(3): # 多次运行以计算平均值
                 result = gpu_intensive_task(a, b)

        print("Time Stats:", get_all_time_logger_stats(called_only=True))
        print("Memory Stats:", get_all_memory_logger_stats(called_only=True))
        print("-" * 20)

        reset_all_time_logger_stats()
        reset_all_memory_logger_stats()
        print("\n--- All statistics have been reset ---")
        print("Time Stats:", get_all_time_logger_stats())
        print("Memory Stats:", get_all_memory_logger_stats())