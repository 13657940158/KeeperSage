#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSE 后端服务器（纯标准库，无第三方依赖）
  GET /sse/stream?count=10&interval=0.5   SSE 流式接口
  GET /api/ping                           普通 JSON 接口（对照测试用）
运行: python backend_server.py
"""

import json
import time
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOST = "0.0.0.0"
PORT = 8000


class SSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---------- 工具 ----------
    def _cors(self):
        # 允许页面跨域直连（方式B）
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # ---------- 路由 ----------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/sse/stream":
            self.handle_stream(parsed)
        elif parsed.path == "/api/ping":
            self._send_json({"ok": True, "msg": "pong", "ts": time.time()})
        else:
            self._send_json({"ok": False, "msg": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ---------- SSE ----------
    def handle_stream(self, parsed):
        qs = parse_qs(parsed.query)
        try:
            total = max(1, min(int(qs.get("count", ["10"])[0]), 1000))
            interval = max(0.05, float(qs.get("interval", ["0.5"])[0]))
        except ValueError:
            total, interval = 10, 0.5

        # 浏览器断线重连时会自动带 Last-Event-ID，据此实现"断点续传"
        last_id = self.headers.get("Last-Event-ID")
        start = int(last_id) + 1 if (last_id and last_id.isdigit()) else 0

        # ===== SSE 响应头（关键）=====
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")  # 必须
        self.send_header("Cache-Control", "no-cache, no-transform")           # 禁止缓存
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")                           # 防 nginx 等反代缓冲
        self._cors()
        self.end_headers()
        self.close_connection = True  # 流结束后直接关闭连接

        def push(event=None, data=None, event_id=None, retry=None):
            """按 SSE 协议拼一条事件并立即 flush（不 flush 就没有流式效果）"""
            lines = []
            if retry is not None:
                lines.append(f"retry: {retry}")          # 告诉浏览器重连间隔(ms)
            if event_id is not None:
                lines.append(f"id: {event_id}")          # 断点续传依据
            if event:
                lines.append(f"event: {event}")          # 不带则为默认 message 事件
            text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            for line in str(text).split("\n"):           # 多行数据必须拆成多个 data:
                lines.append(f"data: {line}")
            self.wfile.write(("\n".join(lines) + "\n\n").encode("utf-8"))
            self.wfile.flush()                           # ★ 每条立即推送

        try:
            # ":" 开头的行是注释，客户端会忽略，常用于心跳保活
            self.wfile.write(b": welcome, stream opened\n")
            self.wfile.flush()

            push(event="meta", retry=3000,
                 data={"server": "backend_server.py", "total": total, "resumeFrom": start})

            if start > 0:
                print(f"[resume] Last-Event-ID={last_id}，从第 {start} 条续传")

            for i in range(start, total):
                push(event_id=i, data={                  # 不带 event 字段 → 前端 onmessage
                    "index": i, "total": total,
                    "value": random.randint(0, 100),
                    "time": time.strftime("%H:%M:%S"),
                })
                time.sleep(interval)

            push(event="multi-line", data="第一行\n第二行\n第三行")   # 多行数据演示
            push(event="done", data={"msg": "全部发送完毕", "total": total})
        except (BrokenPipeError, ConnectionResetError):
            print(f"[disconnect] 客户端中断，停止推送 (path={self.path})")

    def log_message(self, fmt, *args):
        print(f"[backend] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    # 线程服务器：SSE 是长连接，必须多线程才能同时服务页面和多个流
    server = ThreadingHTTPServer((HOST, PORT), SSEHandler)
    server.daemon_threads = True
    print(f"SSE backend: http://127.0.0.1:{PORT}")
    print("  SSE : /sse/stream?count=10&interval=0.5")
    print("  JSON: /api/ping")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")