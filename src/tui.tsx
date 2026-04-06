import * as dotenv from "dotenv";

dotenv.config();

import { Box, render, Text, useApp, useStdout } from "ink";
import BigText from "ink-big-text";
import Gradient from "ink-gradient";
import Spinner from "ink-spinner";
import TextInput from "ink-text-input";
import React, { useEffect, useRef, useState } from "react";
import { ChatAgent, getProvider } from "./index";
import { initLogger, logger } from "./logger";

const Header = () => {
  return (
    <Box flexDirection="column" width="100%" marginBottom={1}>
      <Box flexDirection="row" width="100%" justifyContent="center">
        <Gradient name="cristal">
          <BigText text="SUPPPORTER" font="tiny" />
        </Gradient>
      </Box>
    </Box>
  );
};

const Message = React.memo(
  ({
    role,
    text,
    model,
    duration,
  }: {
    role: string;
    text: string;
    model?: string;
    duration?: number;
  }) => {
    const isUser = role === "user";
    return (
      <Box
        width="100%"
        alignSelf={isUser ? "flex-end" : "flex-start"}
        flexDirection={isUser ? "row-reverse" : "row"}
        alignItems="flex-end"
        marginBottom={1}
      >
        <Box
          borderStyle="round"
          borderColor={isUser ? "green" : "blue"}
          paddingX={1}
          maxWidth="70%"
        >
          <Text>{text}</Text>
        </Box>
        {!isUser && (model || duration !== undefined) && (
          <Box marginLeft={1}>
            <Text color="gray" dimColor italic>
              ({model}
              {duration !== undefined ? ` in ${duration.toFixed(2)}s` : ""})
            </Text>
          </Box>
        )}
      </Box>
    );
  },
);

const App = () => {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<
    { role: string; text: string; model?: string; duration?: number }[]
  >([]);
  const [isThinking, setIsThinking] = useState(false);
  const [dimensions, setDimensions] = useState({
    columns: stdout.columns,
    rows: stdout.rows,
  });
  const agentRef = useRef<ChatAgent | null>(null);

  useEffect(() => {
    const onResize = () => {
      setDimensions({
        columns: stdout.columns,
        rows: stdout.rows,
      });
    };

    stdout.on("resize", onResize);
    return () => {
      stdout.off("resize", onResize);
    };
  }, [stdout]);

  useEffect(() => {
    try {
      const provider = getProvider();

      const tools = [
        {
          functionDeclarations: [
            {
              name: "get_current_time",
              description: "Get the current system time",
              parameters: { type: "OBJECT" as const, properties: {} },
            },
          ],
        },
      ];

      const registry = {
        get_current_time: () => ({ time: new Date().toLocaleTimeString() }),
      };

      agentRef.current = new ChatAgent(provider, {
        tools,
        registry,
        systemInstruction:
          "You are a helpful assistant. Be concise and professional.",
      });
    } catch (error) {
      logger.error`Initialization Error: ${error instanceof Error ? error.stack : String(error)}`;
      exit();
    }
  }, [exit]);

  const handleSubmit = async () => {
    if (!input.trim() || isThinking) return;

    const userText = input.trim();
    setInput("");

    if (userText.toLowerCase() === "/exit") {
      logger.info("User requested exit");
      exit();
      return;
    }

    if (userText.toLowerCase() === "/clear") {
      agentRef.current?.clearHistory();
      setHistory([]);
      return;
    }

    setHistory((prev) => [...prev, { role: "user", text: userText }]);
    setIsThinking(true);
    const startTime = performance.now();

    try {
      const response = await agentRef.current?.execute(userText);
      const endTime = performance.now();
      const totalDuration = (endTime - startTime) / 1000;

      setHistory((prev) => [
        ...prev,
        {
          role: "agent",
          text: response?.text || "",
          model: response?.model,
          duration: totalDuration,
        },
      ]);
    } catch (error) {
      logger.error`Error executing agent: ${error instanceof Error ? error.stack : String(error)}`;
      setHistory((prev) => [
        ...prev,
        {
          role: "agent",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <Box
      flexDirection="column"
      height={dimensions.rows}
      width={dimensions.columns}
      padding={0}
      overflowY="hidden"
    >
      <Header />

      <Box flexDirection="column" flexGrow={1} flexBasis={0} overflowY="hidden">
        {history.map((msg, i) => (
          <Message
            // biome-ignore lint/suspicious/noArrayIndexKey: history is only ever appended
            key={`${msg.role}-${i}`}
            role={msg.role}
            text={msg.text}
            model={msg.model}
            duration={msg.duration}
          />
        ))}
        {isThinking && (
          <Box alignSelf="flex-start" marginLeft={2}>
            <Text color="yellow">
              <Spinner type="dots" /> Thinking...
            </Text>
          </Box>
        )}
      </Box>

      <Box
        flexDirection="row"
        borderStyle="single"
        borderColor="magenta"
        paddingX={1}
      >
        <Text color="green" bold>
          ❯{" "}
        </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder="Type a message..."
        />
        {!input && (
          <Box marginLeft={1}>
            <Text color="gray" dimColor>
              (/exit to quit, /clear to reset)
            </Text>
          </Box>
        )}
      </Box>
    </Box>
  );
};

console.clear();
initLogger().then(() => {
  logger.info("Starting Suppporter TUI");
  render(<App />);
});
