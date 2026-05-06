# Server-Sent Events

Server-Sent Events (SSE) is a standard for pushing a stream of messages from a server to a
client over a single long-lived HTTP response. The response uses the content type
`text/event-stream` and stays open while the server writes events.

## Wire format

Each event is a block of `field: value` lines followed by a blank line. The `data` field
carries the payload, the optional `event` field names the event type, and the optional `id`
field lets a reconnecting browser send a `Last-Event-ID` header so the server can resume.
Lines starting with a colon are comments and are often sent as keep-alive pings so that
proxies do not close an idle connection.

## Why SSE suits token streaming

Language models produce output token by token. Streaming those tokens to the client as
they arrive lowers the time until the first words appear, even though the total generation
time is unchanged. SSE is a natural fit: the traffic is one-directional, it runs over plain
HTTP, and it passes through most proxies and load balancers.

A streaming RAG endpoint commonly sends one event with the retrieved sources first, then
one event per generated token, and finally a closing event that marks the end of the
answer.

## SSE compared with WebSockets

WebSockets provide a full-duplex channel and need an upgrade handshake. They are the
better choice when the client must send many messages during the same connection. For a
request followed by a streamed response, SSE is simpler. The browser `EventSource` API only
issues GET requests, so clients that need to POST a JSON body read the stream with `fetch`
instead.
