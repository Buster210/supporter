class ToolError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message
