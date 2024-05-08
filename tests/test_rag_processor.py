import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("world")

# 第一次调用 asyncio.run()
asyncio.run(main())

# 尝试第二次调用 asyncio.run()，这将会导致 RuntimeError
asyncio.run(main())
