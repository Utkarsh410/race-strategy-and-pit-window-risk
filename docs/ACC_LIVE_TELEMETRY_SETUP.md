# ACC live telemetry: setup and first-session test

RaceSim can receive **read-only** live timing data from Assetto Corsa Competizione
(ACC). This is phase 1: it establishes and validates the connection only. Nothing
received here feeds the strategy engine yet.

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
