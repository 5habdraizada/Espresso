"""PackML state machine for the cell (project brief, weeks 6-7). TODO: build this.

Suggested shape:
  * State enum below, plus a table of allowed transitions, e.g.
        IDLE --start--> STARTING --done--> EXECUTE --hold--> HOLDING --done--> HELD
        HELD --unhold--> UNHOLDING --done--> EXECUTE
        any --stop--> STOPPING --done--> STOPPED --reset--> RESETTING --done--> IDLE
        any --abort (e-stop)--> ABORTING --done--> ABORTED --clear--> CLEARING --done--> STOPPED
  * EXECUTE runs one cycle at a time (see runner.run_cycles) and checks for commands between them.
  * A StopRequested from a driver, or a failed pick, becomes a hold/abort command, never a crash.
  * Every state change is logged and published on OPC UA for the HMI.

Test it on the sim first: make SimRobot raise StopRequested mid-move (OQ-03) and delete a
part just before the grasp (OQ-04); the cell must recover without restarting the program.
"""

from __future__ import annotations

from enum import Enum


class State(Enum):
    IDLE = "Idle"
    STARTING = "Starting"
    EXECUTE = "Execute"
    COMPLETING = "Completing"
    COMPLETE = "Complete"
    HOLDING = "Holding"
    HELD = "Held"
    UNHOLDING = "Unholding"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    RESETTING = "Resetting"
    ABORTING = "Aborting"
    ABORTED = "Aborted"
    CLEARING = "Clearing"
