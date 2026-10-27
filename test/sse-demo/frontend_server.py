#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端服务器（纯标准库）：
  1. 托管本目录静态文件（index.html）       → http://127.0.0.1:8080
  2. /backend/xxx 反向代理到 127.0.0.1:8000/xxx
     —— 逐字节透传、绝不攒包，页面用同源相对路径订阅 SSE，天然无跨域
运行: python frontend_server.py
"""

import http.client
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 8080
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
PROXY_PREFIX = "/backend"

# 代理时不透传的逐跳（hop-by-hop）头，长度/分块由代理重新决定
HOP_HEADERS = {"transfer-encoding", "content-length", "connection", "keep-alive"}


class FrontendHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        if self.path.startswith(PROXY_PREFIX):
            return self.proxy()
        return super().do_GET()

    def proxy(self):
        target = self.path[len(PROXY_PREFIX):] or "/"
        conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=300)
        headers_sent = False
        try:
            fwd = {}
            for h in ("Last-Event-ID", "Accept", "Cache-Control"):  # 重连 ID 必须透传
                v = self.headers.get(h)
                if v:
                    fwd[h] = v
            conn.request("GET", target, headers=fwd)
            resp = conn.getresponse()

            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            self.close_connection = True

            # ★ 关键：read1 = 上游来一点就转发一点，绝不能 read(大数) 攒包
            while True:
                chunk = resp.read1(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            print("[proxy] 客户端断开")
        except Exception as e:
            print(f"[proxy] 错误: {e}")
            if not headers_sent:          # 头还没发出去时才能返回 502
                try:
                    self.send_error(502, f"backend unavailable: {e}")
                except Exception:
                    pass
        finally:
            conn.close()

    def log_message(self, fmt, *args):
        print(f"[frontend] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = ThreadingHTTPServer((HOST, PORT), FrontendHandler)
    server.daemon_threads = True
    print(f"frontend: http://127.0.0.1:{PORT}  (静态 + 代理 /backend/* -> :{BACKEND_PORT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")