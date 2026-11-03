# ACC live telemetry: setup and first-session test

RaceSim can receive **read-only** live timing data from Assetto Corsa Competizione
(ACC). The receiver validates the connection; the **ACC Performance** page analyses laps,
stints and pace and runs strategy analysis on demand (see
[Live performance analysis](#live-performance-analysis)). RaceSim only reads: it never
sends commands to the game.

```
Windows PC (Ethernet)                                      MacBook (Wi-Fi)
┌───────────────────────────────────────────┐   UDP 9100   ┌───────────────────────────────────┐
│ ACC ──UDP 9000 (localhost)── bridge ──────┼──────────────┼──▶ receiver ──▶ health / log      │
│      broadcasting API     acc_windows_    │ signed JSON  │    acc_mac_receiver.py or the     │
│      (register + passwd)  bridge.py       │              │    dashboard page "ACC Live       │
└───────────────────────────────────────────┘              │    Telemetry"                     │
                                                           └───────────────────────────────────┘
```

* ACC's broadcasting API is **not** a plain UDP stream. A client must register
  with ACC (protocol version 4) using the connection password from ACC's
  `broadcasting.json`. Only then does ACC send realtime messages, and only to
  that client. The bridge therefore runs **on the Windows PC**, receives the
  native ACC messages on localhost, converts them to RaceSim's JSON schema and
  forwards them to the Mac.
* Two different ports:
  * **9000** (example): ACC's own listener (`updListenerPort`). Only the bridge
    uses it, on the Windows PC.
  * **9100** (example): RaceSim's forwarding port. The bridge sends to it and
    the Mac receiver listens on it.
* The bridge is read-only. It registers with an **empty command password** and
  only ever sends REGISTER, UNREGISTER, REQUEST_ENTRY_LIST and
  REQUEST_TRACK_DATA. It never sends camera, HUD or replay commands, and it never
  reads Windows shared memory.

## Protocol sources and verification status

ACC's broadcasting protocol is documented by the C# SDK that Kunos ships with
the game (`<ACC install folder>\sdk\broadcasting\`). It is not published online.
The implementation in `src/racesim/integrations/acc/protocol.py` was
cross-checked against two independent open-source implementations:

* [bradfier/acbc](https://github.com/bradfier/acbc): Rust, **AGPL-3.0**. Its
  wire-format facts and its published register test vector were used. **No code
  was copied.**
* [geniusdex/broawp `accbroadcast`](https://pkg.go.dev/github.com/geniusdex/broawp/accbroadcast):
  Go, BSD-3-Clause, at commit `d7f5ba008f03`.

The `broadcasting.json` fields (`updListenerPort`, `connectionPassword`,
`commandPassword`) come from community guides: the
[ACC Race Control wiki](https://github.com/LennysLounge/ACC-Race-Control/wiki)
and the [Racelab docs](https://docs.racelab.app/docs/first-steps/acc).

| Element | Status |
|---|---|
| Little-endian encoding, `u16`-length UTF-8 strings, protocol version 4 | verified in both sources |
| REGISTER (version, display name, connection password, update interval in ms, command password) | verified in both sources; byte-exact against acbc's test vector |
| UNREGISTER, REQUEST_ENTRY_LIST | verified |
| REQUEST_TRACK_DATA (type 11 + connection id) | **unverified layout**; disable with `request_track_data: false` or `--no-track-data-request` |
| REGISTRATION_RESULT, REALTIME_UPDATE, REALTIME_CAR_UPDATE, ENTRY_LIST, ENTRY_LIST_CAR, lap info | verified in both sources |
| TRACK_DATA, BROADCASTING_EVENT | one source (acbc) |
| Replay block in REALTIME_UPDATE, gear encoding, signed vs unsigned temperatures, meaning of `sessionEndTime` | the sources disagree or are silent. Replay updates are flagged, gear is not used, temperatures are forwarded raw, and `sessionEndTime` is forwarded as `session_end_time_raw_s`, **not** as remaining time |

**Unit tests do not prove a live connection.** They use hand-built datagrams and
a fake ACC endpoint on loopback. Only the first-session test below verifies the
real integration.

## What the UDP broadcasting API provides

**Provided:**

* session type and phase
* session time
* track name and length
* air and track temperature
* raw cloud, rain and wetness bytes
* for each car: position, laps, last, best and current lap times, splits,
  pit-lane location, speed, spline position and delta
* entry list (race number, car model, drivers)
* events (lap completed, penalties, accidents)

**Not provided:** physics channels (fuel, tyre temperatures, pressures and wear,
throttle, brake, steering). These need the Windows shared-memory interface,
which is out of scope for this phase. The schema keeps optional slots for them,
and they are always empty from `acc_udp`. ACC only sends data while a session is
loaded (driving or spectating), not in the menus.

## Windows PC (runs ACC and the bridge)

1. **Python 3.11 or newer** from python.org. Tick "Add python.exe to PATH".
2. **Copy the project** to the PC (for example `git clone` or a zip). Open
   PowerShell in the project folder and install only the bridge's dependencies:

   ```bash
   python -m venv .venv-bridge
   ```

   ```bash
   .venv-bridge\Scripts\activate
   ```

   ```bash
   pip install -r requirements-acc-bridge.txt
   ```

   The bridge needs only pydantic and PyYAML. RaceSim's numerical stack is not
   required on Windows.
3. **Find ACC's configuration:**
   `C:\Users\<you>\Documents\Assetto Corsa Competizione\Config\broadcasting.json`.
4. **Back it up first:**

   ```bash
   copy "$env:USERPROFILE\Documents\Assetto Corsa Competizione\Config\broadcasting.json" "$env:USERPROFILE\Documents\broadcasting.json.bak"
   ```

5. **Edit only the needed fields** in Notepad, keeping any other existing
   settings. It should contain:

   ```json
   {
       "updListenerPort": 9000,
       "connectionPassword": "<choose-a-password>",
       "commandPassword": ""
   }
   ```

   Save with Notepad's default "Save" so the file's existing encoding is kept.
   Some community reports say ACC's config files are UTF-16; this has not been
   verified for `broadcasting.json`, so don't change the encoding. Restart ACC
   after editing.
6. **Set the secrets** in the same PowerShell window. Never put them in YAML:

   ```bash
   $env:ACC_CONNECTION_PASSWORD = "<the connectionPassword from step 5>"
   ```

   ```bash
   $env:RACESIM_BRIDGE_SECRET = "<the same long random secret you will set on the Mac>"
   ```

   Generate a secret with
   `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
7. **Windows Ethernet address** (for `allowed_sender_ips` on the Mac): run
   `ipconfig` and look for "IPv4 Address" under the Ethernet adapter.
8. **Configure the Mac address:** copy `config\acc_bridge.example.yaml` to
   `config\acc_bridge.yaml` and set `acc_bridge.target_host` to the **Mac's
   Wi-Fi IPv4 address**, not the Windows address. Or pass `--target-host` on the
   command line.
9. **Start the bridge:**

   ```bash
   python scripts\acc_windows_bridge.py --config config\acc_bridge.yaml --target-host <MAC_WIFI_IP>
   ```

10. **Start ACC and load a session**, for example a practice session on track.
    The bridge log should show:
    `Registered with ACC (connection id N)`, then `Track: ...`, then
    `Session PRACTICE / phase SESSION`. A status line appears every 10 s.
11. **Logs:** the console by default; add `--log-file bridge.log` to keep a
    file. Passwords and the secret are never logged.
12. **Troubleshooting:**
    * `ACC rejected registration: ...`: `ACC_CONNECTION_PASSWORD` does not match
      `connectionPassword` in `broadcasting.json`.
    * `no registration result from ACC`: no session is loaded, or
      `updListenerPort` differs from `--acc-port`. Restart ACC after editing
      `broadcasting.json`.
    * `ACC not reachable on 127.0.0.1:9000`: ACC is not running or is not
      listening on that port.
    * If registration works but nothing else arrives, try
      `--no-track-data-request` (the unverified message).
    * `Could not decode ACC message`: possible protocol change in a newer game
      version. Run with `--log-level DEBUG`, keep the log, and stop using the
      data until it's resolved.

## MacBook (runs the receiver and the dashboard)

1. **Activate the existing environment:**

   ```bash
   cd race-strategy-and-pit-window-risk
   ```

   ```bash
   source .venv/bin/activate
   ```

   No new dependencies are needed (pydantic and PyYAML are already installed).
2. **Find the Mac's Wi-Fi IPv4 address:**

   ```bash
   ipconfig getifaddr en0
   ```

3. **Set the same shared secret:**

   ```bash
   export RACESIM_BRIDGE_SECRET='<same secret as on Windows>'
   ```

4. **Start the receiver** on the forwarding port. It listens on loopback only
   unless you bind it explicitly:

   ```bash
   python scripts/acc_mac_receiver.py --host 0.0.0.0 --port 9100 --csv
   ```

   Use the Wi-Fi IP instead of `0.0.0.0` to listen only on Wi-Fi. Optionally set
   `allowed_sender_ips: [<WINDOWS_ETHERNET_IP>]` in `config/acc_bridge.yaml`.
5. **Or use the dashboard.** Start Streamlit from a shell where the secret is
   exported, then open the **ACC Live Telemetry** page in the sidebar. Set the
   bind address and port, then press **Start**:

   ```bash
   streamlit run app.py
   ```

   Only one process can listen on a port, so run either the script or the
   dashboard receiver, not both.
6. **Verify packets:** the status line or the dashboard should move from
   `WAITING_FOR_DATA` to `CONNECTED`, with receiver, bridge, ACC and telemetry
   each shown separately. `invalid` should stay at 0. A growing
   `authentication_failed` count means the two secrets differ.
7. **Logs:** `outputs/acc_sessions/acc_<UTC time>_<id>.jsonl`, one file per run,
   never overwritten. With `--csv` a CSV copy is also written. The dashboard's
   **Replay** mode reads these files, and labels them REPLAY.

## Local mock test (no ACC needed)

```bash
python scripts/acc_mac_receiver.py --insecure-no-auth --status-interval 1 --duration 20
```

In a second terminal:

```bash
python scripts/send_mock_acc_telemetry.py --count 20 --interval 0.5 --insecure-no-auth
```

Expected:

* `CONNECTED` with `MOCK track=MOCK TRACK (synthetic)` and `last_seq`
  increasing with no gaps;
* `STALE` about 3 s after the sender stops and `DISCONNECTED` after 10 s;
* a JSONL log path printed on exit.

Mock data is always labelled MOCK and never shown as live ACC telemetry.

## Networking checklist

* Both machines are on the same trusted local network. Ethernet and Wi-Fi on the
  same router work, provided the router's **client/AP isolation** is off.
* The bridge's `target_host` is the **Mac's Wi-Fi IPv4 address**, which can
  change after a reconnect (re-check with `ipconfig getifaddr en0`).
* **Mac firewall** (System Settings → Network → Firewall): allow incoming
  connections for the Python process running the receiver or Streamlit.
* **Windows Firewall:** the bridge only *sends* outbound UDP to the Mac and talks
  to ACC on localhost, so no inbound rule is normally needed on Windows. Allow
  Python if Windows prompts.
* A failed `ping` doesn't prove UDP is blocked; many devices drop ICMP. Use the
  mock sender from Windows to test the path (send to the Mac's IP with the
  secret set):

  ```bash
  python scripts\send_mock_acc_telemetry.py --host <MAC_WIFI_IP> --port 9100 --count 10
  ```

* Never forward the receiver port on your router or expose it to the internet.
  The HMAC authenticates the sender on a trusted LAN. It does not encrypt
  anything and isn't internet-grade security.

## First live session test

1. **Mac:** export the secret, then start the receiver:

   ```bash
   python scripts/acc_mac_receiver.py --host 0.0.0.0 --port 9100 --csv
   ```

2. **Windows:** set `ACC_CONNECTION_PASSWORD` and `RACESIM_BRIDGE_SECRET`, run
   the mock sender command above against the Mac, and confirm the Mac shows
   `MOCK` packets. This tests the network path.
3. **Windows:** start the bridge with `--target-host <MAC_WIFI_IP>`, then start
   ACC and drive out in a practice session.
4. **Check on the Mac:**
   * `bridge=CONNECTED acc=CONNECTED telemetry=CONNECTED`, with no MOCK label;
   * the real track name;
   * the position and lap counter changing.
5. Complete two laps. Confirm `last=` matches ACC's lap timing, then compare a
   `LAP_COMPLETED` event in the log with the game.
6. Go back to the ACC menu. Within about 10 s the bridge should log
   `No ACC data ... re-registering`, and the Mac should show ACC as
   `WAITING_FOR_DATA` or `STALE`.
7. Keep both logs. If any value looks wrong, report it with the JSONL file; the
   `field_sources` entry says which native message each value came from.

## Live performance analysis

Open **ACC Performance** in the Streamlit sidebar. It shares the receiver with the *ACC Live
Telemetry* page, so changing page or rerunning never starts a second one. To analyse a recorded
log, choose *Replay* on that page or run:

```bash
python scripts/analyse_acc_session.py outputs/acc_sessions/acc_<stamp>_<id>.jsonl
```

**Update order.** The bridge now forwards extra optional fields, and an older receiver rejects
them. Update the Mac first, or both together. A new receiver works with an old bridge, but
sector times stay UNAVAILABLE until the Windows bridge is updated.

### Behaviour measured on real ACC sessions

These come from recorded sessions at Circuit Paul Ricard (practice, 15 minutes), not from the
protocol documentation. Race, qualifying and multi-stint behaviour has not been recorded yet.

| Observation | How RaceSim handles it |
|---|---|
| A session restart keeps the bridge's session id, and the session clock returns to 0 | The clock regression starts a new session; the bridge now issues a new id as well |
| The out-lap crosses the line without changing `laps` or `lastLap` | Pit evidence comes from the lap's own time window, so the first timed lap is not called an out-lap |
| `elapsed + sessionEndTime` stays at 900.0 during `SESSION` | Time remaining is DERIVED from it, only in `SESSION` and only while the sum stays stable |
| In `PRE_FORMATION`, `sessionEndTime` falls while elapsed stays 0 | Time remaining is UNAVAILABLE in that phase |
| Broadcasting events repeat many times | Events are de-duplicated and never count laps |
| The lap timer keeps running across a restart | Lap times come from `lastLap`, never from the running timer |

### Lap classification

| Class | Meaning | In pace statistics |
|---|---|---|
| `CLEAN_CANDIDATE` | ACC says valid, on track throughout, telemetry covers the lap | Yes |
| `INVALID` | ACC's `isInvalid` flag | No |
| `INCOMPLETE` | No lap time could be established | No |
| `OUT_LAP` / `IN_LAP` | Pit activity inside the lap's window, or ACC's flag | No |
| `UNCERTAIN` | Evidence is missing: validity unknown, telemetry gap over 5 s, not green, race start lap, or over 7% from the median with no known cause | No |

Slow laps are never discarded: they stay in every table and export. A pit visit starts a new stint
but does not prove a tyre change, and leaving the garage at the start is not a stint change.

### Channels and where they come from

| Channel | Status |
|---|---|
| Track, session type and phase, position, laps, last and best lap, car location, speed, temperatures | LIVE MEASUREMENT over UDP |
| Time remaining, current lap, stint number, consumption, lap classification | DERIVED FROM TELEMETRY |
| Sector times | Needs the updated bridge. Accepted only if the sectors add up to the lap; the per-sector meaning is not yet confirmed on real data |
| Out-lap and in-lap flags | Needs the updated bridge; not yet confirmed on real data |
| Fuel, tyre compound, age, pressures, temperatures, wear | **UNAVAILABLE.** Not in the UDP broadcasting API; needs a future Windows shared-memory reader |

The analysis code for fuel and tyres is in place and tested with fixtures. A shared-memory reader
must run on Windows and forward `fuel_kg` in kilograms (ACC reports litres, so it must convert),
`tyre_compound`, `tyre_age_laps`, `tyre_pressures_psi` and `tyre_temperatures_c`.

### Strategy analysis

Runs only when you press the button, never per packet. Unknown values stay blank and are never
filled from the synthetic configuration.

| Readiness | Meaning |
|---|---|
| `BLOCKED` | A required input is missing or the data are stale. The model does not run |
| `SCENARIO_ONLY` | The model runs on stated assumptions. Candidates and windows appear, with no pit recommendation |
| `RECOMMENDATION_ELIGIBLE` | Tyres and fuel measured, live `RACE` in `SESSION`, 5 or more clean laps, tyre model declared calibrated |

Over UDP alone, fuel and tyres are not measured, so `RECOMMENDATION_ELIGIBLE` cannot be reached
until a shared-memory reader exists. The default tyre model is synthetic, so treat predicted times
as illustrative. Monte Carlo is not included, because `run_monte_carlo` simulates a whole race and
cannot start from a mid-race state.

### Outputs

All are written next to the session log in `outputs/acc_sessions/` and never overwrite earlier files:

* `<log>.jsonl`: normalised telemetry, plus `session_metadata`, `completed_lap`, `stint_summary`, `data_quality` and `strategy_snapshot` records.
* `<log>_laps.csv` and `<log>_stints.csv`, written when the receiver stops.
* `<log>_summary.html`: post-session report.

To try the page without ACC (labelled MOCK, and never eligible for a recommendation), use enough
samples per lap that the mock clock has no gaps:

```bash
python scripts/send_mock_acc_telemetry.py --host 127.0.0.1 --port 9100 --insecure-no-auth --count 300 --interval 0.05 --lap-samples 30
```


## Calibrating RaceSim from your recorded sessions

`scripts/calibrate_from_acc.py` fits RaceSim's model to your own ACC laps. It
replays every recorded log through the same lap tracker the dashboard uses,
keeps only real (`acc_udp`) sessions, and converts the completed laps to
RaceSim's lap CSV schema. For each track and parameter it then reports either
an estimate or how much more data is needed. Nothing is applied to the
configuration automatically.

```bash
python scripts/calibrate_from_acc.py
```

```bash
python scripts/calibrate_from_acc.py --track "Circuit Paul Ricard" --compound DRY --fuel-start 60 --fuel-per-lap 2.9
```

Outputs (in `outputs/`):

* `acc_calibration_laps.csv`: the lap data. You can also use it with
  `scripts/validate_model.py --data ...` or upload it in dashboard Section F.
* `acc_calibration_sufficiency.csv`: data needed versus available, per parameter.
* `acc_calibration_report.md`: the full report.

| Parameter (per track) | Minimum data | How to get it |
|---|---|---|
| Observed clean pace | 5 clean laps | any practice run |
| Lap-to-lap variability | 8 clean laps in stints of 4 or more | runs of 4+ consecutive clean laps |
| Pit loss (cross-check) | 1 observed stop + 5 clean laps | drive through the pits once, with an in-lap and an out-lap |
| Tyre degradation | 12 clean laps with known tyre age **and a declared compound** | long runs that start from the pits; pass `--compound` |
| Held-out validation | 30 training + 30 held-out clean laps, in several separate stints | several sessions or stints at the same track |

Assumptions the calibration makes:

* The UDP feed has **no tyre compound and no fuel**. `--compound` and
  `--fuel-start`/`--fuel-per-lap` are your declarations and are recorded as
  assumptions.
* Without fuel data the degradation slope is reported as *confounded with fuel
  burn* and marked unreliable. It is never corrected with RaceSim's synthetic
  default consumption rate.
* Tyre age counts laps since the stint began, only for stints that start in the
  pits (a fresh set is assumed). For stints first seen on track it is unknown.
* Only laps the tracker classifies as clean are used for fitting. Invalid laps,
  out-laps, in-laps and uncertain laps are kept in the CSV but excluded.
* These are recorded **simulator** sessions. The results calibrate RaceSim to
  ACC, not to real-world racing.

## Shared-memory physics (Windows, optional)

The UDP broadcasting API has no physics, fuel or tyre channels. ACC also
publishes them through Windows **shared memory** (`Local\acpmf_physics`,
`acpmf_graphics`, `acpmf_static`). Shared memory can only be read on the PC
running ACC, so the bridge reads it there and forwards selected samples to the
Mac as a new `physics` message type.

```
ACC ─ shared memory ─▶ reader thread (20 Hz, memory only) ─▶ newest sample ─▶ forwarder thread (5 Hz) ─┐
ACC ─ UDP broadcast ─▶ broadcast client (main thread) ─────────────────────────────────────────────────┴▶ BridgeSender ─▶ Mac
```

* **Library:** [`pyaccsharedmemory`](https://github.com/rrennoir/PyAccSharedMemory)
  1.0.0 (MIT, released 2022-09-19, Windows only). RaceSim was checked against
  its source (`src/pyaccsharedmemory.py`): `accSharedMemory().read_shared_memory()`
  and `close()`, and the `Physics` / `Graphics` / `Static` attribute names.
* **Library behaviours handled:**
  * Its "unchanged frame" check compares only `suspension_travel`, so it drops
    every sample while the car is stationary. RaceSim disables that check and
    de-duplicates on ACC's physics `packetId` instead.
  * An unknown enum value makes the library raise for the whole read. RaceSim
    counts it as a read error and keeps going.
* **Units:** from Kunos's *ACC Shared Memory Documentation v1.8.12* (in the
  library repository). Values are forwarded **in native units**. Every channel
  has a `unit_status`:
  * `kunos`: the unit is stated by Kunos.
  * `conflict`: the sources disagree. For example, Kunos says `fuel` is kg, the
    library README says litres, and Kunos gives fuel-per-lap in litres.
  * `unstated`: the expected unit is shown but not verified, for example tyre
    pressure in psi or temperatures in °C.
  * `library`: stated only by the library README.
* **Compatibility:**
  * Physics is **off by default**; enable it with `--physics` or
    `physics.enabled: true`.
  * Broadcast-only setups are unchanged, and heartbeats are untouched.
  * Update the **Mac receiver first**: an older receiver rejects `physics`
    messages but keeps receiving telemetry.
* **Health:** the Mac reports a separate *physics* component, using the reader
  status sent from Windows:
  * `NOT_STARTED`: no physics messages received yet;
  * `WAITING_FOR_DATA`: ACC is off, in the menu, paused or in a replay;
  * `CONNECTED`: new samples are arriving;
  * `STALE` / `DISCONNECTED`: physics messages have stopped arriving;
  * `ERROR`: shared memory is unavailable or the reader reports an error.

  The overall broadcast status is computed exactly as before.
* **Privacy:** player names, other cars' coordinates and fields Kunos marks as
  unused are not read.
* **Mock data:** a mock physics source can never pass as live. If the envelope
  is live and the payload says mock (or the reverse), the receiver rejects it.

### Windows: install, verify locally, then forward

1. Install the optional dependency in the bridge's environment:

   ```bash
   pip install -r requirements-acc-bridge.txt
   ```

   This installs `pyaccsharedmemory==1.0.0` on Windows only. The library's
   classifiers list Python 3.8–3.10; it has not yet been checked on the newer
   Python the bridge uses, which step 2 verifies.
2. **Verify locally first (no network):** start ACC, load a session, and run:

   ```bash
   python scripts\acc_shared_memory_diagnostic.py
   ```

   It prints every channel with its value, unit and unit status, plus a list of
   missing channels, once per second. Check:
   * `reader status` is `LIVE` and the `packet` number increases;
   * `speed_kmh`, `rpm`, `throttle` and `brake` respond to your inputs;
   * compare `fuel_remaining` with the in-game fuel (this settles the kg/litres
     question), `tyre_pressure` with the in-game pressures, and
     `tyre_core_temp` / `brake_temp` with the in-game temperatures;
   * write down how `gear_raw` maps to the gears shown in the game.

   For a single JSON sample:

   ```bash
   python scripts\acc_shared_memory_diagnostic.py --once --json
   ```

3. **Then enable forwarding:**

   ```bash
   python scripts\acc_windows_bridge.py --target-host <MAC_WIFI_IP> --physics
   ```

   Add `--physics-hz 10` to change the rate. The bridge status line adds the
   physics reader state, samples read, samples sent and read errors.

### macOS: receive and display

1. Run the receiver (or use the dashboard page *ACC Live Telemetry*):

   ```bash
   python scripts/acc_mac_receiver.py --host 0.0.0.0 --port 9100
   ```

   The status line adds `physics=<state> reader=<Windows reader status>`, plus
   speed, rpm, raw fuel and tyre pressures.
2. On the dashboard, the **Physics - shared memory** panel lists every channel
   as LIVE - MEASURED, DERIVED (calculated on the Mac: `speed_ms`,
   `tyre_core_temp_spread`, `tyre_pressure_spread`), UNAVAILABLE, STALE or
   MOCK.
3. **Mock test without ACC:**

   ```bash
   python scripts/send_mock_acc_telemetry.py --physics --insecure-no-auth
   ```

   Pair it with `acc_mac_receiver.py --insecure-no-auth`.

Physics messages are about 2.8 KB each at 5 Hz, roughly 50 MB per hour in the
session log.

### Testing status

* **Tested without ACC (automated):**
  * channel extraction and missing-channel detection;
  * the library workaround, tested against a fake module that reproduces its
    behaviour;
  * the reader's de-duplication, stale handling and error handling;
  * forwarding, UDP reception and health states;
  * clean shutdown of the bridge's threads;
  * rejection of mock data presented as live;
  * the dashboard labels, and the diagnostic command in `--mock` mode.
* **Not yet tested:** reading real shared memory on Windows with ACC running,
  the actual units (fuel, pressures, temperatures, gear encoding), library
  behaviour on Python 3.11+, and the end-to-end path from Windows to the Mac
  dashboard. Follow the three steps above in order and record the results.

### Channels and units

| Channel | ACC field | Unit | Unit status | Meaning |
|---|---|---|---|---|
| `throttle` | physics.gas | fraction 0-1 | kunos | Gas pedal input |
| `brake` | physics.brake | fraction 0-1 | kunos | Brake pedal input |
| `clutch` | physics.clutch | fraction 0-1 | kunos | Clutch pedal input |
| `steering` | physics.steerAngle | -1 to 1 | conflict | Steering input; Kunos: -1.0 to 1.0, library README: 0.0 to 1.0 |
| `gear_raw` | physics.gear | raw integer | unstated | Current gear; encoding not stated (AC convention 0=R, 1=N, 2=1st is unverified for ACC) |
| `rpm` | physics.rpm | rpm | kunos | Engine revolutions per minute |
| `speed_kmh` | physics.speedKmh | km/h | kunos | Car speed |
| `water_temp` | physics.waterTemp | degC (expected) | unstated | Water temperature |
| `turbo_boost` | physics.turboBoost | raw | unstated | Car turbo level |
| `fuel_remaining` | physics.fuel | kg (Kunos) / litres (library README) | conflict | Fuel remaining; Kunos says kg, library README says litres - check against the in-game fuel display |
| `fuel_per_lap_l` | graphics.fuelXLap | litres/lap | kunos | Average fuel consumed per lap |
| `fuel_estimated_laps` | graphics.fuelEstimatedLaps | laps | kunos | Laps possible with current fuel |
| `used_fuel` | graphics.usedFuel | litres (expected) | unstated | Fuel used since last refuelling |
| `max_fuel` | static.maxFuel | litres (expected) | unstated | Maximum fuel tank capacity |
| `velocity` | physics.velocity | m/s (expected) | unstated | Velocity vector, global coordinates |
| `local_velocity` | physics.localVelocity | m/s (expected) | unstated | Velocity vector, car coordinates |
| `acceleration_g` | physics.accG | G (expected) | unstated | Acceleration vector, global coordinates |
| `local_angular_velocity` | physics.localAngularVel | rad/s (expected) | unstated | Angular velocity, car coordinates |
| `heading` | physics.heading | rad (expected) | unstated | Car yaw orientation |
| `pitch` | physics.pitch | rad (expected) | unstated | Car pitch orientation |
| `roll` | physics.roll | rad (expected) | unstated | Car roll orientation |
| `wheel_slip` | physics.wheelSlip | dimensionless (expected) | unstated | Tyre slip |
| `slip_ratio` | physics.slipRatio | "radians" per Kunos (dimensionless expected) | conflict | Tyre slip ratio; Kunos labels it 'in radians', which is unusual for a ratio |
| `slip_angle` | physics.slipAngle | rad (expected) | unstated | Tyre slip angle |
| `wheel_angular_speed` | physics.wheelAngularSpeed | rad/s | kunos | Wheel angular speed |
| `tyre_pressure` | physics.wheelPressure | psi (expected) | unstated | Tyre pressure |
| `tyre_core_temp` | physics.TyreCoreTemp | degC (expected) | unstated | Tyre rubber core temperature |
| `brake_temp` | physics.brakeTemp | degC (expected) | unstated | Brake disc temperature |
| `brake_pressure` | physics.brakePressure | raw (car-specific dash coefficient, Kunos appendix) | kunos | Brake pressure |
| `pad_life` | physics.padLife | mm (library README: starts at 29 mm) | library | Brake pad wear |
| `disc_life` | physics.discLife | mm (library README: starts at 32 mm) | library | Brake disc wear |
| `suspension_travel` | physics.suspensionTravel | m (expected) | unstated | Suspension travel |
| `suspension_damage` | physics.suspensionDamage | raw | unstated | Suspension damage level |
| `tc_in_action` | physics.tc | raw | unstated | Traction control in action |
| `abs_in_action` | physics.abs | raw | unstated | ABS in action |
| `brake_bias` | physics.brakeBias | raw (Kunos appendix 4) | kunos | Front brake bias |
| `pit_limiter_on` | physics.pitLimiterOn | - | kunos | Pit limiter on |
| `is_ai_controlled` | physics.isAIControlled | - | kunos | Car is controlled by the AI |
| `air_temp` | physics.airTemp | degC (expected) | unstated | Air temperature |
| `road_temp` | physics.roadTemp | degC (expected) | unstated | Road temperature |
| `status` | graphics.status | - | kunos | ACC_STATUS (OFF/REPLAY/LIVE/PAUSE) |
| `session_type` | graphics.session | - | kunos | ACC_SESSION_TYPE |
| `completed_laps` | graphics.completedLaps | laps | kunos | Completed laps |
| `position` | graphics.position | - | kunos | Current player position |
| `current_lap_time_ms` | graphics.iCurrentTime | ms | kunos | Current lap time |
| `last_lap_time_ms` | graphics.iLastTime | ms | kunos | Last lap time |
| `best_lap_time_ms` | graphics.iBestTime | ms | kunos | Best lap time |
| `current_sector_index` | graphics.currentSectorIndex | - | kunos | Current track sector |
| `is_in_pit` | graphics.isInPit | - | kunos | Car is pitting |
| `is_in_pit_lane` | graphics.isInPitLane | - | kunos | Car is in pit lane |
| `tyre_compound` | graphics.tyreCompound | - | kunos | Tyre compound used |
| `current_tyre_set` | graphics.currentTyreSet | - | kunos | Tyre set currently in use |
| `track_grip_status` | graphics.trackGripStatus | - | kunos | ACC_TRACK_GRIP_STATUS |
| `rain_intensity` | graphics.rainIntensity | - | kunos | ACC_RAIN_INTENSITY |
| `flag` | graphics.flag | - | kunos | ACC_FLAG_TYPE |
| `session_time_left` | graphics.sessionTimeLeft | ms (expected) | unstated | Session time left |
| `car_model` | static.carModel | - | kunos | Player car model (Kunos id) |
| `track` | static.track | - | kunos | Track name |
| `max_rpm` | static.maxRpm | rpm | kunos | Maximum rpm |
| `sm_version` | static.smVersion | - | kunos | Shared memory version |
| `ac_version` | static.acVersion | - | kunos | ACC version |
