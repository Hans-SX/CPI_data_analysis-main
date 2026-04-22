import numpy as np
from abc import ABC, abstractmethod
from itertools import cycle


class MotionPatterns(ABC):
    @abstractmethod
    def _generate(self) -> np.array:
        pass
    
    @abstractmethod
    def pos_frames(self) -> dict:
        pass

class BigStepForward_SmallStepBack(MotionPatterns):
    def __init__(self, start:float, end:float, interval:float=0.5, pattern = np.array((2, -1))):
        self.start = start
        self.end = end
        self.interval = interval
        self.pattern = cycle(pattern)

        if start < 0 or start > end or end > 17:
            raise ValueError("The range of M-231.17 actuator is 0 - 17 mm.")
            
    def _generate(self):
        current = self.start
        pos = []
        while current <= self.end:
            pos.append(current)
            current = current + next(self.pattern) * self.interval
        
        pos = np.asarray(pos)
        if sum(pos < self.start) > 0 or sum(pos > self.end) > 0:
            raise ValueError("The pattern exceed range 0 - 17 mm.")
        return pos
    
    def pos_frames(self, interval=0.5, time_interval=100):
        pfdict = dict()
        pfdict['pos'] = self._generate()
        total_length = sum(abs(pfdict['pos'][1:] - pfdict['pos'][:-1]))
        time = int(total_length / interval)
        pfdict['frames'] = time * time_interval
        return pfdict
    


def target_axials(x, n, stepsize):
    """
    x: the expected position
    n: 2*n + 1 will be the total examined positions
    """
    left_values = [x - i * stepsize for i in range(1, n+1)]
    right_values = [x + i * stepsize for i in range(1, n+1)]
    return left_values[::-1] + [x] + right_values

def mean_positions_per_second(shift, positions, focal_mpl=8.8, speed=0.5, dt=1.0):
    """
    For a sequence of waypoints, return the mean position of the platform
    over each dt-second interval, assuming constant speed between waypoints.
    Turning points (waypoints) are NOT included in the output.

    Parameters
    ----------
    positions : array-like
        1D array of target positions along the axis.
    focal_mpl : float, default 8.8
        Focal plane position in mm.
    speed : float
        Speed in mm/s (same units as positions).
    dt : float, default 1.0
        Time interval in seconds for which the mean position is computed.

    Returns
    -------
    np.ndarray
        1D array of mean positions over each dt-second interval.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 1 or len(positions) < 2:
        raise ValueError("positions must be 1D with at least 2 points")
    if speed <= 0 or dt <= 0:
        raise ValueError("speed and dt must be > 0")

    res = []

    for p0, p1 in zip(positions[:-1], positions[1:]):
        # Segment length and travel time
        dist = abs(p1 - p0)
        if dist == 0:
            continue
        seg_time = dist / speed

        # Split segment into 1‑second intervals (or dt‑second intervals)
        n_steps = int(seg_time // dt)
        if n_steps == 0:
            continue

        # Direction sign
        sgn = 1 if p1 > p0 else -1

        # For each dt‑second interval, compute the mean position over that interval
        for i in range(n_steps):
            t_start = i * dt
            t_end   = (i + 1) * dt
            # positions at the start and end of the interval (within this segment)
            x_start = p0 + sgn * (t_start / seg_time) * dist
            x_end   = p0 + sgn * (t_end   / seg_time) * dist
            # mean over the interval
            x_mean = 0.5 * (x_start + x_end)
            res.append(x_mean)

        expect_ref = np.array(res, dtype=float) - focal_mpl
        try_ref_to = [target_axials(x, 10, 0.01) for x in expect_ref]
        # In interpolation, there is no pixel shifting.
        # try_shifts = [[shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]

    return try_ref_to, expect_ref