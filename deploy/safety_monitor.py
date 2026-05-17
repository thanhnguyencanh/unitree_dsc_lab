"""Safety monitor — watchdog + e-stop hook for real-robot runs.

Pre-flight (§15.3):
  1. Robot on gantry, EMO in reach.
  2. Policy in damping mode for 30 s.
  3. Low-gain track (50% Kp), 1 m flat walk.
  4. One-step climb on 12 cm box.
  5. Five-step indoor staircase.
  6. Outdoor long staircase only after 1-5 succeed 3 runs in a row.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Implement state-machine + LowState watchdog.")


if __name__ == "__main__":
    main()
