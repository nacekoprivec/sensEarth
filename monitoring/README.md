# Monitoring for SensEarth

This module provides a watchdog client for monitoring system components, scrapers, and services in the SensEarth platform. It allows emitting heartbeats, events, metrics, and registering components to track health, performance, and issues.

## Overview

The monitoring system helps ensure reliability by:
- Tracking component uptime and status.
- Logging events like successes, failures, and errors.
- Recording metrics for performance analysis.
- Providing observability for debugging and alerting.

## Setup

1. Copy `client.py` to your project directory.
2. Set the `MONITORING_API` environment variable to the monitoring API URL (e.g., `http://localhost:8001`).
3. Import the required functions:
   ```python
   from monitoring.client import emit_component_registration, emit_event, emit_metric, emit_heartbeat
   ```

## Usage

### Component Registration

Register a component before emitting events, metrics, or heartbeats. This associates all subsequent emissions with the component.

```python
emit_component_registration(
    name="middleware",
    instance_id="default",
    component_type="middleware"
)
```

### Emitting Events

Log events such as successes, failures, or custom occurrences.

```python
emit_event(
    name="middleware",
    instance_id="default",
    event_type="registering_model",
    severity="INFO",
    message="Register model endpoint called",
    metadata={"model_name": "anomaly_detector"}
)
```

- `severity`: "INFO", "WARNING", "ERROR", etc.
- `metadata`: Optional dict for additional context.

### Emitting Metrics

Record numerical metrics like counts, rates, or timings.

```python
emit_metric(
    name="middleware",
    instance_id="default",
    metric_name="registered_nodes",
    value=42,
    unit="count"
)
```

- `unit`: Optional string like "count", "seconds", "%".

### Emitting Heartbeats

Send periodic status updates to indicate component health.

```python
emit_heartbeat(
    name="middleware",
    instance_id="default",
    status="OK"
)
```

- `status`: "OK", "FAIL", or custom status.
- `metadata`: Optional dict for extra info.

All emissions are sent asynchronously in background threads to avoid blocking your code.

All endpoints can be found in MONITORING_API docs (eg. http://localhost:8001/docs#)