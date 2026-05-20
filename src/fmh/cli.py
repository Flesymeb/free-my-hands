from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import uvicorn

from fmh.approval import decide_review
from fmh.codex_auditor import CodexReviewAuditor
from fmh.config import load_config
from fmh.feishu import FeishuOpenAPIClient, make_feishu_client
from fmh.models import EventSource, Requester, SourceType
from fmh.orchestrator import DeploymentOrchestrator
from fmh.operator_review import make_error_review, make_reuse_plan_review, review_card, review_result_card
from fmh.parser import parse_deployment_request
from fmh.poller import FeishuPollingWorker
from fmh.reusable_workers import (
    build_new_worker_row_plan,
    build_reconnect_plan,
    build_reusable_deployment_plan,
    parse_deployed_models_table,
)
from fmh.runner import make_runner
from fmh.store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="fmh")
    parser.add_argument("--config", default=None, help="Path to config TOML.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--reload", action="store_true")

    parse_cmd = subparsers.add_parser("parse", help="Parse a deployment request.")
    _add_text_args(parse_cmd)

    submit = subparsers.add_parser("submit", help="Submit a deployment request synchronously.")
    _add_text_args(submit)
    submit.add_argument("--source-ref", default="cli")
    submit.add_argument("--user-id", default="cli")

    list_cmd = subparsers.add_parser("list", help="List recent deployment requests.")
    list_cmd.add_argument("--limit", type=int, default=20)

    show = subparsers.add_parser("show", help="Show a deployment request and events.")
    show.add_argument("request_id")

    subparsers.add_parser("list-chats", help="List Feishu chats the bot has joined.")

    send_test = subparsers.add_parser("send-test", help="Send a Feishu test message.")
    target = send_test.add_mutually_exclusive_group()
    target.add_argument("--chat-id", help="Feishu chat_id, usually starts with oc_.")
    target.add_argument("--open-id", help="Feishu open_id for private message.")
    send_test.add_argument("--text", default="free-my-hands test message")
    send_test.add_argument("--card", action="store_true", help="Send an interactive card.")

    read_doc = subparsers.add_parser("read-doc", help="Read Feishu doc/wiki content as Markdown.")
    read_doc.add_argument("doc_token", help="Docx doc_token or wiki node token.")
    read_doc.add_argument("--limit", type=int, default=1200, help="Characters to print.")

    workers = subparsers.add_parser("workers", help="List reusable deployed-model rows.")
    workers.add_argument("--json", action="store_true", help="Print JSON instead of a table.")

    plan_reuse = subparsers.add_parser("reuse-plan", help="Plan deployment onto an existing worker.")
    plan_reuse.add_argument("weight_path", help="Raw model path from a Feishu task/subtask.")
    plan_reuse.add_argument("--gpu-count", type=int, default=0, help="Require at least this many GPUs.")

    reconnect = subparsers.add_parser("reconnect-plan", help="Plan reconnecting a disconnected worker tmux session.")
    reconnect.add_argument("target", help="Worker IP, table row index, or model id.")
    reconnect.add_argument("--history-lines", type=int, default=300)

    new_worker = subparsers.add_parser("new-worker-row", help="Plan the table row for a newly allocated worker.")
    new_worker.add_argument("weight_path", help="Raw model path from a Feishu task/subtask.")
    new_worker.add_argument("--ip", required=True, help="New worker IP address.")
    new_worker.add_argument("--gpu-count", type=int, required=True)
    new_worker.add_argument("--rlaunch-command", default="")

    review_reuse = subparsers.add_parser("review-reuse", help="Create a Codex/operator review packet for a reuse plan.")
    review_reuse.add_argument("weight_path")
    review_reuse.add_argument("--gpu-count", type=int, default=0)
    review_reuse.add_argument("--send-card", action="store_true")

    review_error = subparsers.add_parser("review-error", help="Create a Codex/operator review packet for an error.")
    review_error.add_argument("--stage", required=True)
    review_error.add_argument("--subject-id", required=True)
    review_error.add_argument("--error", required=True)
    review_error.add_argument("--send-card", action="store_true")

    reviews = subparsers.add_parser("reviews", help="List operator review packets.")
    reviews.add_argument("--limit", type=int, default=20)
    reviews.add_argument("--status", default=None)

    review_show = subparsers.add_parser("review-show", help="Show an operator review packet.")
    review_show.add_argument("review_id")
    review_show.add_argument("--prompt-only", action="store_true")

    review_decide = subparsers.add_parser("review-decide", help="Record a manual review decision.")
    review_decide.add_argument("review_id")
    review_decide.add_argument("decision", choices=["APPROVE", "BLOCK", "RETRY", "REQUEST_INFO"])
    review_decide.add_argument("--actor", default="cli")
    review_decide.add_argument("--note", default="")
    review_decide.add_argument("--send-card", action="store_true")

    review_auditor = subparsers.add_parser("review-auditor", help="Run the Codex review auditor.")
    review_auditor.add_argument("--once", action="store_true")
    review_auditor.add_argument("--interval-sec", type=int)

    poll = subparsers.add_parser("poll", help="Poll Feishu for deployment tasks.")
    poll.add_argument("--once", action="store_true", help="Run one polling tick and exit.")
    poll.add_argument("--lookback-sec", type=int, help="For this run, scan recent history.")
    poll.add_argument("--interval-sec", type=int, help="Override polling interval.")
    poll.add_argument("--chat-id", action="append", default=[], help="Override/append Feishu chat_id.")
    poll.add_argument("--document-id", action="append", default=[], help="Override/append Feishu docx document_id.")

    args = parser.parse_args()

    if args.command == "serve":
        if args.config:
            import os

            os.environ["FMH_CONFIG"] = args.config
        uvicorn.run("fmh.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    config = load_config(args.config)
    store = StateStore(config.storage.sqlite_path)

    if args.command == "parse":
        request = parse_deployment_request(_manual_source(_read_text(args)))
        print(json.dumps(request.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "submit":
        source = _manual_source(_read_text(args), args.source_ref, args.user_id)
        request = parse_deployment_request(source)
        orchestrator = DeploymentOrchestrator(
            config=config,
            store=store,
            runner=make_runner(config.runner),
            notifier=make_feishu_client(config.feishu),
        )
        result = orchestrator.submit(request)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "list":
        data = [request.to_dict() for request in store.list_requests(args.limit)]
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if args.command == "show":
        request = store.get_request(args.request_id)
        if request is None:
            raise SystemExit(f"request not found: {args.request_id}")
        data = {
            "deployment": request.to_dict(),
            "events": [event.to_dict() for event in store.events_for(args.request_id)],
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if args.command == "list-chats":
        client = make_feishu_client(config.feishu)
        chats = client.list_chats()
        if not chats:
            print("No chats found. Add the bot to a group and grant im:chat:readonly if needed.")
            return
        print(f"{'chat_id':<40} name")
        for chat in chats:
            print(f"{chat.get('chat_id', ''):<40} {chat.get('name', '')}")
        return

    if args.command == "send-test":
        client = make_feishu_client(config.feishu)
        if args.open_id:
            if args.card:
                client.send_private_card(args.open_id, _test_card(args.text))
            else:
                client.send_private_text(args.open_id, args.text)
            print("sent private test message")
            return
        chat_id = args.chat_id or config.feishu.default_chat_id
        if not chat_id:
            raise SystemExit("missing --chat-id or [feishu].default_chat_id")
        if args.card:
            client.send_chat_card(chat_id, _test_card(args.text))
        else:
            client.send_chat_text(chat_id, args.text)
        print("sent chat test message")
        return

    if args.command == "read-doc":
        client = FeishuOpenAPIClient(config.feishu)
        content = client.get_doc_markdown(args.doc_token)
        print(content[: args.limit])
        if len(content) > args.limit:
            print(f"\n... truncated {len(content) - args.limit} chars")
        return

    if args.command == "workers":
        client = FeishuOpenAPIClient(config.feishu)
        content = client.get_doc_markdown(config.reusable_workers.deployed_models_doc_token)
        rows = parse_deployed_models_table(content)
        if args.json:
            print(json.dumps([row.to_dict(config.reusable_workers) for row in rows], ensure_ascii=False, indent=2))
            return
        print(f"{'row':<4} {'reuse':<5} {'gpu':<3} {'ip':<16} {'model_id':<24} tested")
        for row in rows:
            reusable = "yes" if row.is_reusable(config.reusable_workers) else "no"
            tested = row.tested_tasks.replace("\n", ", ")
            print(f"{row.row_index:<4} {reusable:<5} {row.gpu_count:<3} {row.ip:<16} {row.model_id:<24} {tested}")
        return

    if args.command == "reuse-plan":
        client = FeishuOpenAPIClient(config.feishu)
        content = client.get_doc_markdown(config.reusable_workers.deployed_models_doc_token)
        plan = build_reusable_deployment_plan(
            content,
            args.weight_path,
            config.reusable_workers,
            required_gpu_count=args.gpu_count,
        )
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "reconnect-plan":
        client = FeishuOpenAPIClient(config.feishu)
        content = client.get_doc_markdown(config.reusable_workers.deployed_models_doc_token)
        rows = parse_deployed_models_table(content)
        row = _find_worker_row(rows, args.target)
        if row is None:
            raise SystemExit(f"worker target not found: {args.target}")
        session = f"ssh_{row.gpu_count}_gpu_{row.ip.split('.')[-2]}_{row.ip.split('.')[-1]}"
        windows = _remote_tmux_windows(config.reusable_workers.dev_host, session)
        pane_history = _remote_tmux_history(config.reusable_workers.dev_host, session, args.history_lines)
        plan = build_reconnect_plan(row, config.reusable_workers, windows=windows, pane_history=pane_history)
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "new-worker-row":
        plan = build_new_worker_row_plan(
            args.weight_path,
            args.ip,
            args.gpu_count,
            config.reusable_workers,
            rlaunch_command=args.rlaunch_command,
        )
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "review-reuse":
        client = FeishuOpenAPIClient(config.feishu)
        content = client.get_doc_markdown(config.reusable_workers.deployed_models_doc_token)
        rows = parse_deployed_models_table(content)
        plan = build_reusable_deployment_plan(
            content,
            args.weight_path,
            config.reusable_workers,
            required_gpu_count=args.gpu_count,
        )
        packet = make_reuse_plan_review(weight_path=args.weight_path, plan=plan, rows=rows)
        store.create_review(packet.review_id, packet.stage.value, packet.subject_id, packet.to_dict())
        if args.send_card:
            _send_review_card(config, packet)
        print(json.dumps(packet.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "review-error":
        packet = make_error_review(stage=args.stage, subject_id=args.subject_id, error=args.error)
        store.create_review(packet.review_id, packet.stage.value, packet.subject_id, packet.to_dict())
        if args.send_card:
            _send_review_card(config, packet)
        print(json.dumps(packet.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "reviews":
        print(json.dumps(store.list_reviews(args.limit, status=args.status), ensure_ascii=False, indent=2))
        return

    if args.command == "review-show":
        review = store.get_review(args.review_id)
        if review is None:
            raise SystemExit(f"review not found: {args.review_id}")
        if args.prompt_only:
            payload = review.get("payload", {})
            if isinstance(payload, dict):
                print(payload.get("codex_prompt", ""))
            return
        print(json.dumps(review, ensure_ascii=False, indent=2))
        return

    if args.command == "review-decide":
        review = decide_review(
            store,
            review_id=args.review_id,
            decision=args.decision,
            actor=args.actor,
            source="cli",
            note=args.note,
        )
        if args.send_card:
            client = make_feishu_client(config.feishu)
            chat_id = config.approval.fallback_chat_id or config.feishu.default_chat_id
            if not chat_id:
                raise SystemExit("missing [approval].fallback_chat_id or [feishu].default_chat_id")
            decision = review.get("decision") if isinstance(review.get("decision"), dict) else {}
            client.send_chat_card(chat_id, review_result_card(review, decision))
        print(json.dumps(review, ensure_ascii=False, indent=2))
        return

    if args.command == "review-auditor":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if args.interval_sec is not None:
            config = replace(
                config,
                codex_review=replace(config.codex_review, interval_sec=args.interval_sec),
            )
        auditor = CodexReviewAuditor(config, store, make_feishu_client(config.feishu))
        if args.once:
            print(json.dumps({"processed": auditor.process_once(wait=True)}, ensure_ascii=False, indent=2))
            auditor.shutdown()
            return
        auditor.run_forever()
        return

    if args.command == "poll":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        polling = config.polling
        if args.interval_sec is not None:
            polling = replace(polling, interval_sec=args.interval_sec)
        if args.chat_id:
            polling = replace(polling, chat_ids=args.chat_id)
        if args.document_id:
            polling = replace(polling, document_ids=args.document_id)
        config = replace(config, polling=polling)
        feishu_client = FeishuOpenAPIClient(config.feishu)
        orchestrator = DeploymentOrchestrator(
            config=config,
            store=store,
            runner=make_runner(config.runner),
            notifier=feishu_client if config.feishu.send_notifications else make_feishu_client(config.feishu),
        )
        worker = FeishuPollingWorker(config, store, feishu_client, orchestrator)
        if args.once:
            stats = worker.poll_once(lookback_sec=args.lookback_sec)
            print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
            return
        worker.run_forever()
        return


def _add_text_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Deployment request text.")
    group.add_argument("--text-file", help="File containing deployment request text.")


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    return args.text


def _manual_source(text: str, source_ref: str = "cli", user_id: str = "cli") -> EventSource:
    return EventSource(
        source_type=SourceType.MANUAL,
        source_ref=source_ref,
        requester=Requester(user_id=user_id),
        text=text,
    )


def _test_card(text: str) -> dict[str, object]:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "free-my-hands test"},
            "template": "blue",
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": text}},
        ],
    }


def _find_worker_row(rows, target: str):
    for row in rows:
        if str(row.row_index) == target or row.ip == target or row.model_id == target:
            return row
    return None


def _remote_tmux_windows(dev_host: str, session: str) -> list[str]:
    command = f"tmux list-windows -t {shlex.quote(session)} -F '#{{window_name}}'"
    result = _remote_shell(dev_host, command)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _remote_tmux_history(dev_host: str, session: str, lines: int) -> str:
    # Prefer the ssh window, then fall back to window 0 if the name differs.
    command = (
        f"tmux capture-pane -pt {shlex.quote(session)}:ssh -S -{lines} 2>/dev/null "
        f"|| tmux capture-pane -pt {shlex.quote(session)}:0 -S -{lines} 2>/dev/null"
    )
    result = _remote_shell(dev_host, command)
    return result.stdout if result.returncode == 0 else ""


def _remote_shell(dev_host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-CAXY",
            dev_host,
            command,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _send_review_card(config, packet) -> None:
    client = make_feishu_client(config.feishu)
    chat_id = config.approval.fallback_chat_id or config.feishu.default_chat_id
    if not chat_id:
        raise SystemExit("missing [approval].fallback_chat_id or [feishu].default_chat_id")
    client.send_chat_card(chat_id, review_card(packet, config.approval))


if __name__ == "__main__":
    main()
