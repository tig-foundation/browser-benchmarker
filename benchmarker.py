from tigpy.data import Benchmark, Proof, asDict, Result
from tigpy.frontiers import randomDifficultyOnFrontier
from tigpy.benchmarker import BaseBenchmarker, BenchmarkParams, QueryData, BenchmarkModules
from typing import List
from datetime import datetime
import asyncio
import random
import numpy as np
import itertools
from js import TIG


API_URL = TIG.API_URL


class BrowserBenchmarker(BaseBenchmarker):
    def __init__(self, api_key: str, player_id: str):
        super().__init__(
            api_url=API_URL, 
            api_key=api_key, 
            player_id=player_id, 
            num_workers=1
        )
        self.status = "Initialising"
        self.data = None
        self.current_benchmark = None
        self.selected_algorithms = {}
        self.pending_benchmarks = {}
        self._num_attempts = 0
    
    async def _onWorkerStartAttempt(self, worker_id: int, nonce: int):
        pass

    async def _onWorkerFinishAttempt(self, worker_id: int, nonce: int):
        self._num_attempts += 1

    async def _onWorkerSolution(self, worker_id: int, nonce: int):
        pass

    async def _onWorkerSolutionId(self, worker_id: int, nonce: int):
        pass

    async def _handleWorkerError(self, worker_id: int, e: Exception):
        pass

    async def _handleBenchmarkerError(self, e: Exception):
        self.status = f"Error: {e}"

    async def _pickBenchmarkParams(self, data: QueryData) -> BenchmarkParams:
        self.status = "Picking parameters"
        # pick challenge with bias towards least percent qualifiers
        percent_qualifiers = next((p.percent_qualifiers for p in data.players if p.id == self._player_id), {})
        p = np.array([
            percent_qualifiers.get(challenge.id, 0)
            for challenge in data.block.config.challenges
        ])
        p = np.max(p) - p + 1e-5
        p = p / np.sum(p)
        random_idx = int(np.random.choice(len(data.block.config.challenges), p=p))
        picked_challenge_id = data.block.config.challenges[random_idx].id
        # pick algorithm that we have selected if valid or default
        picked_algorithm_id = self.selected_algorithms.get(picked_challenge_id, "default")
        for algorithm in data.algorithms:
            if (
                algorithm.id == picked_algorithm_id and 
                algorithm.challenge_id == picked_challenge_id and 
                not algorithm.banned and 
                algorithm.block_pushed <= data.block.height
            ):
                break
        else:
            picked_algorithm_id = "default"
        # acceptable difficulty is any point between 2 an upper and lower frontier
        # if total_num_qualifiers > threshold
        #   uppper = frontier_0 + (frontier_0 - min_difficulty) * (total_num_qualifiers / threshold)
        #   lower = frontier_0
        # else
        #   uppper = frontier_0
        #   lower = frontier_0 + (frontier_0 - min_difficulty) * (total_num_qualifiers / threshold)
        frontier_0 = [
            fp for fp in data.frontier_points
            if fp.frontier_idx == 0 and fp.challenge_id == picked_challenge_id
        ]
        difficulty_parameters = next(c.difficulty_parameters for c in data.block.config.challenges if c.id == picked_challenge_id)
        min_difficulty = [p.min_value for p in difficulty_parameters]
        if len(frontier_0):
            frontier_0_difficulty = randomDifficultyOnFrontier(frontier_0, difficulty_parameters)
            num_qualifiers = sum(
                fp.num_qualifiers for fp in data.frontier_points 
                if fp.frontier_idx is not None and fp.challenge_id == picked_challenge_id
            )
            difficulty_multiplier = min(
                num_qualifiers / data.block.config.difficulty_frontiers.num_qualifiers_threshold, 
                data.block.config.difficulty_frontiers.max_difficulty_multiplier
            )
            random_offset = [
                (p - param.min_value + 1) * (1.0 + random.random() * (difficulty_multiplier - 1.0))
                for p, param in zip(frontier_0_difficulty, difficulty_parameters)
            ]
            picked_difficulty = [
                max(min(int(param.min_value - 1 + np.ceil(o)), param.max_value), param.min_value)
                for param, o in zip(difficulty_parameters, random_offset)
            ]
        else:
            # pick min difficulty
            picked_difficulty = min_difficulty
        return BenchmarkParams(
            challenge_id=picked_challenge_id,
            algorithm_id=picked_algorithm_id,
            difficulty=picked_difficulty,
            duration=60,
        )

    async def _queryData(self) -> QueryData:
        self.status = f"Querying data"
        data = await super()._queryData()
        if len(self.selected_algorithms) == 0:
            for c in data.block.config.challenges:
                self.selected_algorithms[c.id] = "default"
        return data

    async def _setupModules(self, data: QueryData, params: BenchmarkParams) -> BenchmarkModules:
        self.status = f"Setting up modules"
        return await super()._setupModules(data, params)

    async def _doBenchmark(
        self, 
        data: QueryData, 
        params: BenchmarkParams, 
        modules: BenchmarkModules, 
        datetime_end: datetime,
        proofs: List[Proof]
    ):
        self._num_attempts = 0
        self.data = asDict(data)
        self.data["block"]["datetime_added"] = self.data["block"]["datetime_added"].isoformat()
        self.current_benchmark = asDict(params)
        self.current_benchmark["datetime_end"] = datetime_end.astimezone().isoformat()
        task = asyncio.create_task(super()._doBenchmark(data, params, modules, datetime_end, proofs))
        while not task.done():
            self.status = f"Benchmarking... Solved {len(proofs)} out of {self._num_attempts} challenges"
            await asyncio.sleep(1)
        self.status = f"Finished"

    async def _doSubmitBenchmark(self, data: QueryData, params: BenchmarkParams, proofs: List[Proof]) -> Result:
        self.status = f"Submitting benchmark"
        result = await super()._doSubmitBenchmark(data, params, proofs)
        if result.status_code == 200:
            self.pending_benchmarks[result.data.benchmark_id] = asDict(params)
        return result
    
    async def _doSubmitProofs(self, benchmark: Benchmark) -> Result:
        self.status = "Submitting proofs"
        self.pending_benchmarks.pop(benchmark.id, None)
        return await super()._doSubmitProofs(benchmark)

    async def start(self):
        self.status = "Starting benchmarker"
        await super().start()

    async def stop(self):
        self.status = "Stopping benchmarker"
        await super().start()
        self.status = "Stopped"

G = {}

async def getBenchmarkerStatus():
    if (b := G.get("benchmarker")) is None:
        return dict(running=False)
    else:
        return dict(
            status=b.status,
            data=b.data,
            current_benchmark=b.current_benchmark,
            selected_algorithms=b.selected_algorithms,
            pending_benchmarks=b.pending_benchmarks
        )

async def startBenchmarker(player_id, api_key):
    if (b := G.get("benchmarker")) is None:
        b = BrowserBenchmarker(player_id=player_id, api_key=api_key)
        G["benchmarker"] = b
        await b.start()

async def stopBenchmarker():
    if (b := G.pop("benchmarker")) is not None:
        await b.stop()
        
async def selectAlgorithm(challenge_id, algorithm_id):
    if (b := G.get("benchmarker")) is not None:
        b.selected_algorithms.update({challenge_id: algorithm_id})