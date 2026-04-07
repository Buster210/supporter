import asyncio

from supporter.tui import SupporterApp


async def main():
    app = SupporterApp()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        tree = app.tree
        print(tree)
        for node in app.screen.walk_children():
            print(
                f"{node} offsets: {node.region} padding: {node.styles.padding} margin: {node.styles.margin}"
            )


asyncio.run(main())
