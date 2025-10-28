"""
아바타 시스템 성능 테스트
목표: p50 ≤ 200ms, p95 ≤ 400ms
"""
import asyncio
import time
import statistics
from typing import List, Dict
import httpx


class AvatarPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.latencies: List[float] = []
    
    async def test_single_request(self, emotion: str, confidence: float) -> float:
        """단일 요청 지연시간 측정"""
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/map-emotion",
                    json={
                        "emotion": emotion,
                        "confidence": confidence,
                        "risk_level": "low"
                    },
                    timeout=5.0
                )
                
                latency = (time.time() - start_time) * 1000  # ms로 변환
                
                if response.status_code == 200:
                    return latency
                else:
                    print(f"❌ 오류: {response.status_code}")
                    return -1
                    
            except Exception as e:
                print(f"❌ 예외: {e}")
                return -1
    
    async def run_load_test(self, num_requests: int = 100):
        """부하 테스트 실행"""
        print("\n" + "="*60)
        print("🚀 아바타 매핑 성능 테스트 시작")
        print("="*60)
        print(f"📊 총 요청 수: {num_requests}")
        print(f"🎯 목표: p50 ≤ 200ms, p95 ≤ 400ms")
        print("="*60 + "\n")
        
        emotions = ["joy", "sad", "anxiety", "anger", "neutral"]
        tasks = []
        
        for i in range(num_requests):
            emotion = emotions[i % len(emotions)]
            confidence = 0.5 + (i % 50) / 100  # 0.5 ~ 0.99
            tasks.append(self.test_single_request(emotion, confidence))
        
        # 병렬 실행
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 실패한 요청 제외
        self.latencies = [lat for lat in results if lat > 0]
        
        # 통계 계산
        self.print_statistics(total_time, num_requests)
    
    def print_statistics(self, total_time: float, num_requests: int):
        """통계 출력"""
        if not self.latencies:
            print("❌ 모든 요청 실패")
            return
        
        self.latencies.sort()
        
        # 백분위수 계산
        p50 = statistics.median(self.latencies)
        p95 = self.latencies[int(len(self.latencies) * 0.95)]
        p99 = self.latencies[int(len(self.latencies) * 0.99)]
        
        min_lat = min(self.latencies)
        max_lat = max(self.latencies)
        avg_lat = statistics.mean(self.latencies)
        std_lat = statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
        
        success_rate = len(self.latencies) / num_requests * 100
        throughput = len(self.latencies) / total_time
        
        print("\n" + "="*60)
        print("📈 테스트 결과")
        print("="*60)
        print(f"✅ 성공률: {success_rate:.1f}% ({len(self.latencies)}/{num_requests})")
        print(f"⏱️  총 소요시간: {total_time:.2f}초")
        print(f"🔥 처리량: {throughput:.1f} req/s")
        print("\n" + "-"*60)
        print("⏱️  지연시간 통계 (ms)")
        print("-"*60)
        print(f"최소:        {min_lat:.2f} ms")
        print(f"평균:        {avg_lat:.2f} ms ± {std_lat:.2f}")
        print(f"중앙값(p50): {p50:.2f} ms  {'✅' if p50 <= 200 else '❌'} (목표: ≤ 200ms)")
        print(f"p95:         {p95:.2f} ms  {'✅' if p95 <= 400 else '❌'} (목표: ≤ 400ms)")
        print(f"p99:         {p99:.2f} ms")
        print(f"최대:        {max_lat:.2f} ms")
        print("="*60)
        
        # 목표 달성 여부
        if p50 <= 200 and p95 <= 400:
            print("🎉 목표 달성! 성능 요구사항을 만족합니다!")
        else:
            print("⚠️  목표 미달성. 최적화가 필요합니다.")
        
        print("="*60 + "\n")
    
    def generate_histogram(self, bins: int = 10):
        """지연시간 분포 히스토그램"""
        if not self.latencies:
            return
        
        print("\n📊 지연시간 분포 히스토그램")
        print("-"*60)
        
        min_lat = min(self.latencies)
        max_lat = max(self.latencies)
        bin_width = (max_lat - min_lat) / bins
        
        for i in range(bins):
            bin_start = min_lat + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for lat in self.latencies if bin_start <= lat < bin_end)
            
            bar = "█" * int(count / len(self.latencies) * 50)
            print(f"{bin_start:6.1f}-{bin_end:6.1f}ms | {bar} {count}")
        
        print("-"*60 + "\n")


async def main():
    tester = AvatarPerformanceTester()
    
    # 부하 테스트 실행
    await tester.run_load_test(num_requests=100)
    
    # 히스토그램 출력
    tester.generate_histogram()


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   🎭 아바타 상태 매핑 성능 테스트                           ║
    ║   목표: 감정 → 표정 반영 p50 ≤ 200ms, p95 ≤ 400ms         ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
