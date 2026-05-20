def graph_safe_rng_available() -> bool:
    return False


def get_all_rng_states() -> dict:
    return {}


class CudaRNGStatesTracker:
    def __init__(self):
        self.states = {}

    def reset(self) -> None:
        self.states = {}

    def get_states(self) -> dict:
        return self.states

    def set_states(self, states: dict) -> None:
        self.states = states


__all__ = ["CudaRNGStatesTracker", "get_all_rng_states", "graph_safe_rng_available"]
