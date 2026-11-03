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
running ACC, so the bridge reads it there and forwards samples to the Mac over
the **same authenticated UDP channel** (one sender, one HMAC secret, one
sequence counter, one receiver), as the message type `physics`.

```
ACC ─ shared memory ─▶ reader thread (20 Hz, memory only) ─▶ newest sample ─▶ forwarder thread (5 Hz) ─┐
ACC ─ UDP broadcast ─▶ broadcast client (main thread) ─────────────────────────────────────────────────┴▶ BridgeSender ─▶ Mac
```

### Design: why physics is its own message, and where the two are combined

* **Separate message, same channel.** Session/timing data (`telemetry`) and
  physics (`physics`) travel as separate signed messages through the same
  sender. The `telemetry` payload is not extended, because receivers validate
  it strictly (`extra="forbid"`): a physics section inside it would make every
  older receiver reject *all* telemetry. A separate type means an older
  receiver rejects only the physics messages and keeps working.
* **Combined on the Mac.** Each physics sample carries the bridge's current
  `session_id`. The receiver pairs it with the newest session data and keeps
  the two freshnesses apart:
  * `telemetry` health follows the UDP-broadcast stream;
  * `physics` health follows the newest physics sample, timed from when the
    Windows reader **took** it (receive time minus the sample's age at
    sending), not from when it arrived.
* **Independent sources.** The bridge keeps forwarding session data when
  shared memory fails, and keeps sending physics *status* messages (about
  once a second) while there are no new samples. The Mac can therefore tell
  "ACC not in a session" from "bridge down".
* **No duplicate or invalid samples:**
  * The reader de-duplicates on ACC's physics `packetId`.
  * The forwarder sends each sample once.
  * The receiver ignores a sample older than the one it holds.
  * Values outside a channel's plausibility range are listed under `invalid`
    with the reason, and are never forwarded or shown as measurements.
* **Timestamps.** ACC's shared memory has no wall-clock time, only
  `packetId`. Three times are recorded separately:
  * `timestamp_utc`: when the Windows reader took the sample;
  * `sample_age_s`: how old the sample was when it was sent;
  * the Mac's own `received_at_utc`.
* **Schema.** `physics_schema_version` 1.1. It renamed the nine 1.0 channels
  below to the integration's target names and added `invalid`. 1.0 messages,
  logs and config channel lists are still accepted, and are renamed on
  receipt.
* **Target names not used yet:** `fuel_l`. The fuel value stays
  `fuel_remaining` until a live session shows whether ACC reports kg (Kunos)
  or litres (library README); a name must not claim an unverified unit.
  `gear` is ACC's raw gear index, not the gear shown on the dash.

| 1.0 name | 1.1 name |
|---|---|
| `steering` | `steering_input` |
| `gear_raw` | `gear` |
| `acceleration_g` | `acceleration` |
| `local_angular_velocity` | `angular_velocity` |
| `tyre_pressure` | `wheel_pressure` |
| `tyre_core_temp` | `tyre_core_temperature` |
| `brake_temp` | `brake_temperature` |
| `abs_in_action` | `abs_activity` |
| `tc_in_action` | `tc_activity` |

### Library

* [`pyaccsharedmemory`](https://github.com/rrennoir/PyAccSharedMemory) **1.0.0**,
  pinned in `requirements-acc-bridge.txt` (Windows only):
  * MIT licence;
  * pure Python (`mmap` + `struct`), no binaries;
  * classifiers list Python 3.8–3.10.
* **Maintenance (checked 2026-11-03):** the only releases (0.0.1, 0.0.2 and
  1.0.0) were all published on 2022-09-19, with no commits since. Five issues
  are open. One of them (#6) is the packet-id de-duplication bug handled
  below. RaceSim was checked against the library's source
  (`src/pyaccsharedmemory.py`): `accSharedMemory().read_shared_memory()` and
  `close()`, and the `Physics` / `Graphics` / `Static` attribute names
  (`ContactPoint`, `CarDamage`, `Wheels`, `Vector3f`).
* **ACC version:** the struct layout follows Kunos's v1.8.12 document. In
  2022 a user reported physics reading as zero after an ACC update; it turned
  out to be their own error (issue #3). Whether 1.0.0 matches the **current**
  ACC build has not been checked. The diagnostic below shows it: watch the
  `sm_version` / `ac_version` channels and whether the values look right.
* **Library behaviours handled:**
  * Its "unchanged frame" check compares only `suspension_travel` (and never
    updates `last_physicsID`), so it drops every sample while the car is
    stationary. RaceSim clears that check before each read and de-duplicates
    on ACC's physics `packetId` instead.
  * An unknown enum value makes the library raise for the whole read. RaceSim
    counts it as a read error (logged at most once per 100 errors). After 20
    consecutive failures it closes and reopens the shared memory.
  * Strings are NUL-padded; RaceSim strips them.
* **If the library proves incompatible:** the reader depends only on the small
  `SharedMemorySource` protocol (`read()` / `close()`). A replacement reader,
  for example a vendored `ctypes` struct, can be swapped in without touching
  the bridge, the protocol or the Mac.

### Units, ranges and semantics

* Values are forwarded **in native units**. Every channel has a
  `unit_status`:
  * `kunos`: the unit is stated by Kunos;
  * `conflict`: the sources disagree. For example, Kunos says `fuel` is kg, the
    library README says litres, and Kunos gives fuel-per-lap in litres;
  * `unstated`: the expected unit is shown but not verified, for example tyre
    pressure in psi or temperatures in °C;
  * `library`: stated only by the library README.
* **Unavailable vs zero.** An absent value or an empty CSV cell means
  *unavailable*; `0` is a real reading (for example, a stationary car's
  `speed_kmh`).
* **Ranges.** Ranges are set only where the bound holds whatever the
  unverified unit turns out to be: pedals 0–1, steering −1–1, speed 0–500,
  rpm 0–30 000, and so on.
* **Semantics.** Pedals are fractions, not percentages. `steering_input` is a
  normalised input, not a wheel angle in degrees. `abs_activity` /
  `tc_activity` are raw activity levels, not the aid settings.
* **Coordinates.** Vectors are `[x, y, z]`, and per-wheel values are
  `[FL, FR, RL, RR]`. Kunos names a global and a car-local frame but does not
  state the axis orientation of either.

### Health

The Mac reports a separate **physics** component:

* `NOT_STARTED`: no physics messages received yet;
* `WAITING_FOR_DATA`: ACC is off, in the menu, paused or in a replay;
* `CONNECTED`: new samples are fresh;
* `STALE` / `DISCONNECTED`: physics messages have stopped arriving;
* `ERROR`: shared memory is unavailable or the reader reports an error.

The **overall** state accounts for physics once it is expected, that is once
any physics message has arrived in this receiver run. Set
`health.require_physics: true` to expect it from the start. If the session
components are all `CONNECTED` but the expected physics is not, overall is
**`DEGRADED`**: session and timing data are current, physics is not. The
integration is never reported fully healthy in that state. Broadcast-only
setups never send physics and keep the original four-component rule.

On a new ACC session (the telemetry `session_id` changes), the receiver drops
the previous session's physics sample. When the bridge restarts (a new
sender run), the sample held from the old run is dropped too.

### Logging and export

* Every physics message is written to the session JSONL as the raw signed
  envelope plus the Mac receive time and quality flags. Each record keeps:
  * session id, sample sequence and ACC packet ids;
  * the Windows sampling time;
  * the available / missing / invalid channels.
* Quality flags include:
  * `physics_without_session`: physics arrived before any session data;
  * `physics_session_mismatch`;
  * `physics_invalid_channels`;
  * `physics_sample_older`.
* Logs are opened in exclusive-create mode and never overwritten.
* **Physics CSV** (one row per sample; wheels `_fl/_fr/_rl/_rr`, vectors
  `_x/_y/_z`). The export streams the log, so memory use stays flat, and it
  refuses to overwrite an existing CSV.
  * When a receiver run ends: `acc_mac_receiver.py --csv` writes
    `<log>_physics.csv` if physics arrived.
  * From an existing log:

    ```bash
    python scripts/analyse_acc_session.py <log.jsonl> --physics-csv
    ```

    Add `--physics-channels speed_kmh,rpm,wheel_pressure` to choose channels.
* Size: physics messages are about 3 KB each, so at 5 Hz the session log
  grows by roughly 50 MB per hour.

### Windows: install, verify locally, then forward

1. Install the optional dependency in the bridge's existing environment:

   ```bash
   pip install -r requirements-acc-bridge.txt
   ```

   This installs `pyaccsharedmemory==1.0.0` on Windows only. Nothing changes
   on the Mac.
2. **Verify locally first (no network):** start ACC, load a session (driving),
   and run:

   ```bash
   python scripts\acc_shared_memory_diagnostic.py
   ```

   Once per second it prints speed, rpm, gear, throttle, brake, fuel, the four
   tyre pressures and the four tyre core temperatures, in native units.
   `UNAVAILABLE` and `INVALID` are shown explicitly. Check:
   * `reader status` is `LIVE` and the `packet` number increases;
   * speed, rpm, throttle and brake respond to your inputs;
   * fuel matches the in-game fuel. Write down whether it is kg or litres:
     this settles the `fuel_remaining` / `fuel_l` question;
   * the pressures and core temperatures match the in-game tyre display
     (psi? °C?);
   * how `Gear` maps to the gear shown in the game.

   Two more views:
   * every channel with its unit and unit status:

     ```bash
     python scripts\acc_shared_memory_diagnostic.py --full
     ```

   * a single JSON sample:

     ```bash
     python scripts\acc_shared_memory_diagnostic.py --once --json
     ```

3. **Then enable forwarding:**

   ```bash
   python scripts\acc_windows_bridge.py --target-host <MAC_WIFI_IP> --physics
   ```

   Add `--physics-hz 10` to change the rate (`physics.sample_hz` /
   `forward_hz` in the config). The bridge status line adds the physics reader
   state, samples read, samples sent and read errors.

### macOS: receive and display

1. Update the Mac first, or both sides together. A 1.0-only receiver rejects
   1.1 physics messages, but its telemetry is unaffected.
2. Run the receiver, or use the dashboard page *ACC Live Telemetry*:

   ```bash
   python scripts/acc_mac_receiver.py --host 0.0.0.0 --port 9100
   ```

   The status line adds `physics=<state> reader=<Windows reader status>`, plus
   speed, rpm, raw fuel and tyre pressures.
3. On the dashboard, the **Physics - shared memory** panel has tabs:
   * *Driver inputs*;
   * *Vehicle*;
   * *Dynamics*;
   * *Tyres*, a per-wheel table;
   * *Brakes*, per wheel, plus bias, ABS, TC and damage;
   * *Data quality*: health, sample age, measurement time, available / missing
     / invalid channels, reader errors, session, connection status, and the
     units still unverified;
   * *All channels*.

   Values are labelled:
   * **LIVE MEASUREMENT**;
   * **DERIVED**, calculated on the Mac: `speed_ms`,
     `tyre_core_temperature_spread`, `wheel_pressure_spread`;
   * **STALE**;
   * **UNAVAILABLE**;
   * **INVALID**;
   * **MOCK**.
4. **Mock test without ACC:**

   ```bash
   python scripts/send_mock_acc_telemetry.py --physics --insecure-no-auth
   ```

   Pair it with `acc_mac_receiver.py --insecure-no-auth`. The diagnostic also
   has a `--mock` mode that runs on any OS.

### Testing status

Mock tests and real-ACC verification are kept separate. Passing the first says
nothing about the second.

* **Verified with mock / fake sources (automated, 36 physics tests; full suite
  353 passing on macOS):**
  * extraction of every channel kind (scalars, wheels, vectors, contact points,
    damage), and missing-channel detection;
  * range checks, unavailable vs zero, and acceptance plus renaming of schema
    1.0;
  * the library workaround, against a fake module that reproduces its
    behaviour;
  * reader de-duplication, staleness, read errors and reopening after repeated
    failures;
  * forwarding and UDP reception;
  * health: session without physics, stale physics → `DEGRADED`, sample-age
    timing, physics without session, and `require_physics`;
  * session change, bridge restart and older-sample handling;
  * UDP-only backward compatibility;
  * JSONL logging and the physics CSV, including legacy logs and the
    no-overwrite rule;
  * clean shutdown of the bridge's threads, and rejection of mock data
    presented as live;
  * the dashboard rows, and the diagnostic in `--mock` mode. The dashboard
    panel itself was rendered headlessly with mock data (Streamlit `AppTest`),
    not in a browser.
* **Not yet verified (needs the Windows PC with ACC running):**
  * reading real shared memory with `pyaccsharedmemory` 1.0.0 on the bridge's
    Python version and the current ACC build;
  * the actual units (fuel kg/L, pressures, temperatures) and the gear
    encoding;
  * whether the plausibility ranges hold for real data;
  * the end-to-end path from Windows to the Mac dashboard.

  Follow the three Windows steps above in order and record the results. Until
  then the live physics integration is **not** complete.

### Channels and units

| Channel | Group | ACC field | Unit | Unit status | Valid range | Meaning |
|---|---|---|---|---|---|---|
| `throttle` | inputs | physics.gas | fraction 0-1 | kunos | 0 to 1 | Gas pedal input (not a percentage) |
| `brake` | inputs | physics.brake | fraction 0-1 | kunos | 0 to 1 | Brake pedal input (not a percentage) |
| `clutch` | inputs | physics.clutch | fraction 0-1 | kunos | 0 to 1 | Clutch pedal input |
| `steering_input` | inputs | physics.steerAngle | normalised -1 to 1 (not degrees) | conflict | -1 to 1 | Steering input as a fraction of full lock, NOT a wheel angle; Kunos: -1.0 to 1.0, library README: 0.0 to 1.0 |
| `gear` | vehicle | physics.gear | raw ACC gear index | unstated | - | Current gear in ACC's raw encoding, not the displayed gear (AC convention 0=R, 1=N, 2=1st is unverified for ACC) |
| `rpm` | vehicle | physics.rpm | rpm | kunos | 0 to 30000 | Engine revolutions per minute |
| `speed_kmh` | vehicle | physics.speedKmh | km/h | kunos | 0 to 500 | Car speed |
| `water_temp` | vehicle | physics.waterTemp | degC (expected) | unstated | -60 to 250 | Water temperature |
| `turbo_boost` | vehicle | physics.turboBoost | raw | unstated | - | Car turbo level |
| `is_engine_running` | vehicle | physics.isEngineRunning | - | kunos | - | Engine running |
| `pit_limiter_on` | vehicle | physics.pitLimiterOn | - | kunos | - | Pit limiter on |
| `is_ai_controlled` | vehicle | physics.isAIControlled | - | kunos | - | Car is controlled by the AI |
| `fuel_remaining` | vehicle | physics.fuel | kg (Kunos) / litres (library README) | conflict | 0 to 250 | Fuel remaining; Kunos says kg, library README says litres - check against the in-game fuel display |
| `fuel_per_lap_l` | vehicle | graphics.fuelXLap | litres/lap | kunos | 0 to 50 | Average fuel consumed per lap |
| `fuel_estimated_laps` | vehicle | graphics.fuelEstimatedLaps | laps | kunos | 0 to 10000 | Laps possible with current fuel |
| `used_fuel` | vehicle | graphics.usedFuel | litres (expected) | unstated | 0 to 10000 | Fuel used since last refuelling |
| `max_fuel` | vehicle | static.maxFuel | litres (expected) | unstated | 0 to 250 | Maximum fuel tank capacity |
| `velocity` | dynamics | physics.velocity | m/s (expected) | unstated | - | Velocity [x, y, z], global (world) coordinates; axis orientation not stated by Kunos |
| `local_velocity` | dynamics | physics.localVelocity | m/s (expected) | unstated | - | Velocity [x, y, z], car coordinates; axis orientation not stated by Kunos |
| `acceleration` | dynamics | physics.accG | G (expected) | unstated | - | Acceleration [x, y, z], global (world) coordinates; axis orientation not stated by Kunos |
| `angular_velocity` | dynamics | physics.localAngularVel | rad/s (expected) | unstated | - | Angular velocity [x, y, z], car coordinates; axis orientation not stated by Kunos |
| `heading` | dynamics | physics.heading | rad (expected) | unstated | - | Car yaw orientation |
| `pitch` | dynamics | physics.pitch | rad (expected) | unstated | - | Car pitch orientation |
| `roll` | dynamics | physics.roll | rad (expected) | unstated | - | Car roll orientation |
| `wheel_slip` | dynamics | physics.wheelSlip | dimensionless (expected) | unstated | - | Tyre slip per wheel |
| `slip_ratio` | dynamics | physics.slipRatio | "radians" per Kunos (dimensionless expected) | conflict | - | Tyre slip ratio; Kunos labels it 'in radians', which is unusual for a ratio |
| `slip_angle` | dynamics | physics.slipAngle | rad (expected) | unstated | - | Tyre slip angle |
| `wheel_pressure` | tyres | physics.wheelPressure | psi (expected) | unstated | 0 to 100 | Tyre pressure |
| `tyre_core_temperature` | tyres | physics.TyreCoreTemp | degC (expected) | unstated | -60 to 400 | Tyre rubber core temperature |
| `wheel_angular_speed` | tyres | physics.wheelAngularSpeed | rad/s | kunos | - | Wheel angular speed |
| `suspension_travel` | tyres | physics.suspensionTravel | m (expected) | unstated | -1 to 1 | Suspension travel |
| `tyre_contact_point` | tyres | physics.tyreContactPoint | m (expected) | unstated | - | Tyre contact point [x, y, z] per wheel, global (world) coordinates; axis orientation not stated by Kunos |
| `tyre_contact_normal` | tyres | physics.tyreContactNormal | unit vector (expected) | unstated | - | Tyre contact normal [x, y, z] per wheel, global (world) coordinates; axis orientation not stated by Kunos |
| `tyre_contact_heading` | tyres | physics.tyreContactHeading | unit vector (expected) | unstated | - | Tyre contact heading [x, y, z] per wheel, global (world) coordinates; axis orientation not stated by Kunos |
| `brake_pressure` | brakes | physics.brakePressure | raw (car-specific dash coefficient, Kunos appendix) | kunos | - | Brake pressure |
| `brake_temperature` | brakes | physics.brakeTemp | degC (expected) | unstated | -60 to 2000 | Brake disc temperature |
| `brake_bias` | brakes | physics.brakeBias | raw (Kunos appendix 4) | kunos | - | Front brake bias |
| `pad_life` | brakes | physics.padLife | mm (library README: starts at 29 mm) | library | - | Brake pad wear |
| `disc_life` | brakes | physics.discLife | mm (library README: starts at 32 mm) | library | - | Brake disc wear |
| `abs_activity` | brakes | physics.abs | raw | unstated | - | ABS in action (raw level, not a setting) |
| `tc_activity` | brakes | physics.tc | raw | unstated | - | Traction control in action (raw level, not a setting) |
| `car_damage` | car | physics.carDamage | raw | unstated | - | Car damage [front, rear, left, right, center]; library notes 'center' is the sum of all |
| `suspension_damage` | car | physics.suspensionDamage | raw | unstated | - | Suspension damage level |
| `air_temp` | car | physics.airTemp | degC (expected) | unstated | -60 to 100 | Air temperature |
| `road_temp` | car | physics.roadTemp | degC (expected) | unstated | -60 to 100 | Road temperature |
| `status` | session | graphics.status | - | kunos | - | ACC_STATUS (OFF/REPLAY/LIVE/PAUSE) |
| `session_type` | session | graphics.session | - | kunos | - | ACC_SESSION_TYPE |
| `completed_laps` | session | graphics.completedLaps | laps | kunos | 0 to 10000 | Completed laps |
| `position` | session | graphics.position | - | kunos | 0 to 1000 | Current player position |
| `current_lap_time_ms` | session | graphics.iCurrentTime | ms | kunos | - | Current lap time |
| `last_lap_time_ms` | session | graphics.iLastTime | ms | kunos | - | Last lap time |
| `best_lap_time_ms` | session | graphics.iBestTime | ms | kunos | - | Best lap time |
| `current_sector_index` | session | graphics.currentSectorIndex | - | kunos | - | Current track sector |
| `is_in_pit` | session | graphics.isInPit | - | kunos | - | Car is pitting |
| `is_in_pit_lane` | session | graphics.isInPitLane | - | kunos | - | Car is in pit lane |
| `tyre_compound` | tyres | graphics.tyreCompound | - | kunos | - | Tyre compound used |
| `current_tyre_set` | tyres | graphics.currentTyreSet | - | kunos | - | Tyre set currently in use |
| `track_grip_status` | session | graphics.trackGripStatus | - | kunos | - | ACC_TRACK_GRIP_STATUS |
| `rain_intensity` | session | graphics.rainIntensity | - | kunos | - | ACC_RAIN_INTENSITY |
| `flag` | session | graphics.flag | - | kunos | - | ACC_FLAG_TYPE |
| `session_time_left` | session | graphics.sessionTimeLeft | ms (expected) | unstated | - | Session time left |
| `car_model` | session | static.carModel | - | kunos | - | Player car model (Kunos id) |
| `track` | session | static.track | - | kunos | - | Track name |
| `max_rpm` | vehicle | static.maxRpm | rpm | kunos | 0 to 30000 | Maximum rpm |
| `sm_version` | session | static.smVersion | - | kunos | - | Shared memory version |
| `ac_version` | session | static.acVersion | - | kunos | - | ACC version |
