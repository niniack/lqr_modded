"""Wall-clock progress without adding rows to epoch-based quality histories."""

from datetime import datetime, timezone
import math
import time


def _console_line(snapshot):
    parts = []
    if "epoch" in snapshot:
        parts.append(f"epoch {snapshot['epoch']}/{snapshot['epochs']}")
    if "epoch_document_budget" in snapshot:
        parts.append(f"{snapshot.get('epoch_documents', 0):,}/{snapshot['epoch_document_budget']:,} docs")
    elif "probe_documents" in snapshot:
        parts.append(f"probe {snapshot.get('teacher_documents', 0):,}/{snapshot['probe_documents']:,} docs")
    if "total_optimizer_steps" in snapshot:
        parts.append(f"step {snapshot.get('optimizer_steps_completed', 0):,}/{snapshot['total_optimizer_steps']:,}")
    if snapshot.get("batch_loss") is not None:
        parts.append(f"loss {snapshot['batch_loss']:.4g}")
    if "eta_training_seconds" in snapshot:
        hours, minutes = divmod(max(0, math.ceil(snapshot["eta_training_seconds"] / 60)), 60)
        parts.append(f"ETA {hours}h{minutes:02d}m" if hours else f"ETA {minutes}m")
    parts.append(snapshot["phase"].replace("_", " "))
    return " | ".join(parts)


class TrainingProgress:
    """Emit bounded-cadence console lines and a latest-value W&B summary."""

    def __init__(self, interval_seconds: float = 60):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Progress interval must be finite and positive")
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.last_report = float("-inf")
        self.phase = "starting"
        self.fields = {}
        self.counts = {}
        self.seconds = {}
        self.run = None

    def state_dict(self):
        return {
            "elapsed_seconds": time.monotonic() - self.started,
            "phase": self.phase,
            "fields": self.fields.copy(),
            "counts": self.counts.copy(),
            "seconds": self.seconds.copy(),
        }

    def load_state_dict(self, state):
        self.started = time.monotonic() - state["elapsed_seconds"]
        self.last_report = float("-inf")
        self.phase = state["phase"]
        self.fields = state["fields"].copy()
        self.counts = state["counts"].copy()
        self.seconds = state["seconds"].copy()

    def count(self, **increments) -> None:
        for key, value in increments.items():
            self.counts[key] = self.counts.get(key, 0) + value

    def call(self, phase, function, *args, **kwargs):
        """Measure host wall time, including natural device waits; not GPU kernel time."""
        previous = self.phase
        self.phase = phase
        self.report()
        started = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            self.seconds[phase] = self.seconds.get(phase, 0.0) + time.monotonic() - started
            self.phase = previous

    def batches(self, loader):
        iterator = iter(loader)
        while True:
            try:
                batch = self.call("data_wait", next, iterator)
            except StopIteration:
                return
            yield batch

    def record_update(self, elapsed_seconds, examples):
        self.seconds["student_update"] = self.seconds.get("student_update", 0.0) + elapsed_seconds
        self.count(optimizer_steps_completed=1, optimized_documents=examples)

    def attach_run(self, run) -> None:
        self.run = run
        self.report(force=True)

    def report(self, *, force=False, phase=None, **fields) -> None:
        self.phase = phase or self.phase
        self.fields.update(fields)
        now = time.monotonic()
        if not force and now - self.last_report < self.interval_seconds:
            return
        snapshot = self._snapshot(now)
        print(_console_line(snapshot), flush=True)
        if self.run is not None:
            # Heartbeats update summary only; history steps belong to completed epochs.
            self.run.summary["progress"] = snapshot
        self.last_report = now

    def _snapshot(self, now):
        snapshot = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "elapsed_seconds": round(now - self.started, 3),
            **self.fields,
            **self.counts,
            "timing_seconds": {key: round(value, 3) for key, value in self.seconds.items()},
        }
        teacher_docs = self.counts.get("teacher_documents", 0)
        target_seconds = sum(self.seconds.get(key, 0) for key in ("source_read", "tokenize", "teacher_targets"))
        if teacher_docs and target_seconds:
            snapshot["teacher_documents_per_second"] = teacher_docs / target_seconds
        completed_steps = self.counts.get("optimizer_steps_completed", 0)
        update_seconds = self.seconds.get("student_update", 0)
        if completed_steps and update_seconds:
            snapshot["mean_student_update_seconds"] = update_seconds / completed_steps
            total_steps = self.fields.get("total_optimizer_steps")
            total_docs = self.fields.get("total_training_documents")
            if total_steps is not None and total_docs is not None and teacher_docs:
                snapshot["eta_training_seconds"] = (
                    max(0, total_docs - teacher_docs) * target_seconds / teacher_docs
                    + max(0, total_steps - completed_steps) * update_seconds / completed_steps
                )
                snapshot["eta_excludes"] = "Future fixed-probe/validation evaluation and checkpoint writes"
        return snapshot
