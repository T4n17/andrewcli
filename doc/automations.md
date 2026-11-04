# Automations

An **automation** runs a prompt, a workflow, an event, or a built-in command automatically: at startup, at a wall-clock time, on an interval, on a cron schedule, or once at a given date. Each automation is one YAML file in `~/.config/andrewcli/automations/`. Two disabled examples (`daily_brief.yaml`, `nightly_reset.yaml`) are seeded on first run.

```yaml
# ~/.config/andrewcli/automations/daily_brief.yaml
description: Weekday morning briefing
trigger:
  at: "08:30"
  days: weekdays
run: "Give me today's date and the three most important items in todo.md."
```

---

## Fields

| Key | Required | Description |
|-----|----------|-------------|
| `run` | yes | What to execute. Text starting with `/` is a slash command; anything else is a prompt. |
| `trigger` | yes | When to execute. A single trigger, or a list of triggers (the earliest upcoming one wins). |
| `name` | no | Defaults to the file name. Letters, digits, `_`, `-`, `.`; must be unique. |
| `description` | no | Shown in notifications and `/status`. |
| `enabled` | no | Default `true`. Disabled automations can still be run manually with `/automations run`. |
| `domain` | no | A domain name or list. The automation is only armed while one of them is active. Default: every domain. |
| `silent` | no | Default `false`. When `true`, runs are recorded in `/status` but not printed or streamed. |
| `tools` | no | Tool names forced into prompt runs regardless of routing (same as an event's `required_tools`). Also the only tools the `analyze` step may use. |
| `analyze` | no | An instruction for the agent. The output of `run` is sent to the agent with it, and you see the agent's answer instead of the raw output. See below. |

### What `run` can do

`run` is resolved exactly like typed input:

| `run` | Effect |
|-------|--------|
| `"Summarize my inbox"` | Prompt answered by an isolated, memory-free generation (the same path events use). The conversation memory is never touched. |
| `/reset`, `/rag reindex`, `/stop loop`, `/echo Stand up!` | Built-in command (`/echo` prints its text as-is, e.g. for reminders). |
| `/summarize notes.md` | Workflow of the active domain; its streamed output is the automation's response. |
| `/loop "Poll /health until 200" 20` | Starts an event on the bus; the event then runs on its own. |

### Analyzing the output with the agent

With `analyze`, the output of `run` goes to the agent instead of straight to you:

```yaml
trigger: {cron: "0 * * * *"}
run: /monitor_atesaq
analyze: "If the site changed, explain what changed. Otherwise say so in one line."
```

- **Workflows, built-ins, and prompts:** their output is analyzed once per run.
- **Events** (`run: /monitor "..."`): each output the event produces is analyzed as it arrives. An event that never sends anything to the agent has no output to analyze.
- **What you see:** the agent's answer, shown like any automation output. The raw output appears in the `/status [id]` trace as a `run` step, cut to a short preview. With `silent: true`, the answer is only recorded in `/status`.
- **No tools by default:** the output may contain untrusted text such as web pages, so the analysis only has the tools listed under `tools`, and no skills. Listing a tool like `execute_command` lets that text influence what the agent runs.
- **No conversation memory:** the analysis doesn't see your conversation and isn't added to it.
- **Limits:** output past 12,000 characters is cut. Every run calls the model, even when nothing changed, so hourly automations mean hourly model calls. Errors such as an unknown command are shown as they are.

### Triggers

| Trigger | Fires |
|---------|-------|
| `startup` | Once per session, as soon as the automation is first armed. Not repeated on domain switch or reload. |
| `{every: "30m"}` | Every interval after arming. Accepts `90s`, `30m`, `2h`, `1d`, `1h30m`, or a number of seconds. |
| `{at: "08:30"}` | Daily at that time. `at` also accepts a list: `at: ["08:30", "18:00"]`. |
| `{at: "08:30", days: weekdays}` | Only on matching days: `mon`..`sun`, ranges (`mon-fri`), lists, `weekdays`, `weekends`. |
| `{cron: "*/15 9-17 * * mon-fri"}` | Five-field cron (`minute hour day-of-month month day-of-week`) with `*`, lists, ranges, steps, and month/day names. |
| `{once: "2026-12-01 09:00"}` | Once at that datetime. Ignored if the time has already passed. `dd-mm-yyyy-hh-mm` also works. |

Combine triggers with a list:

```yaml
trigger:
  - startup
  - every: 2h
run: /rag reindex
```

> Quote times (`"08:30"`). YAML reads some unquoted times such as `17:45` as base-60 numbers; AndrewCLI converts them back, but quoting avoids surprises.

Times use the local clock. Runs missed while AndrewCLI was not running are not caught up.

---

## Commands

```
/automations                 — list automations with their state and next run
/automations reload          — re-read the files and re-arm (no restart needed)
/automations run [name]      — run one automation now, once
/status                      — automation runs appear next to events (name#N)
/stop [name]                 — disarm an automation until the next reload or domain switch
```

Possible states in `/automations`: `next run …`, `running`, `done` (startup-only or one-time automation that already ran), `disabled`, `inactive in this domain`, `not armed` (added after startup; run `/automations reload`). Files that fail to parse are listed under **Errors** with the reason.

---

## How it works

Automations reuse the event machinery instead of adding a scheduler of their own. Whenever a domain becomes active (at startup and after every domain switch), `AndrewCore.set_domain()` re-arms automations, adding one `AutomationEvent` per automation that applies to that domain to the shared event bus:

- `condition()` sleeps until the next run. It wakes at least every 30 s to re-check the wall clock, so schedules stay correct across system suspend.
- Prompt runs set `message`. Command runs set the generic `Event.command` attribute, and `AndrewCore._event_dispatch` routes those through built-ins, workflows, and events.
- Runs are serialized with user turns through `Domain.busy_lock`, logged for `/status`, and notified like any other event (CLI banner, tray balloon).

Switching domains re-arms automations for the new domain (other running events are unaffected).
