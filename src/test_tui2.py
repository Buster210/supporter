from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static


class TestApp(App):
    CSS = """
    #chat-view {
        height: 1fr;
        padding: 0; margin: 0;
        background: transparent;
        layout: vertical;
        overflow-y: auto; overflow-x: hidden;
        scrollbar-size: 0 0;
    }
    .msg {
        width: 100vw; height: auto;
    }
    .msg.right { align-horizontal: right; }
    .bubble {
        width: auto; max-width: 80%;
        background: green; color: black;
        border: solid green;
    }
    #input {
        border: solid magenta;
        height: 3;
    }
    """
    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="chat-view"):
            msg = Vertical(classes="msg right")
            msg.mount(Static("hi", classes="bubble"))
            yield msg
        yield Static("Input Area", id="input")

app = TestApp()
if __name__ == "__main__":
    app.run()
