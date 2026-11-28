import asyncio
import time

async def task(name, seconds):
    print(f"[{name}] 开始...")
    await asyncio.sleep(seconds)
    print(f"[{name}] 结束！")

async def main():
    start = time.time()
    
    # 1. 同时派发任务进后台（不阻塞）
    t1 = asyncio.create_task(task("任务A", 2))
    t2 = asyncio.create_task(task("任务B", 1))
    
    # 2. 等待所有任务完成
    await t1
    await t2
    
    print(f"总耗时: {time.time() - start:.2f} 秒")  # 总耗时约 2.00 秒

if __name__ == "__main__":
    asyncio.run(main())