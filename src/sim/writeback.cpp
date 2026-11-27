#include "sim/writeback.hpp"

#include "sim/tile_cost.hpp"

namespace tensys {
namespace {

enum class BankState { free, reducing, queued, writing };

struct Bank {
    BankState state{BankState::free};
    TileStep last_step{};
    Count queued_at{};
};

struct ArrayRun {
    TileStep step;
    std::size_t bank;
    Count finish;
};

struct DmaRun {
    OperationKind kind;
    Shape tile;
    std::optional<std::size_t> bank;
    Count finish;
};

void accumulate(Count& total, Count value) {
    total = checked_add(total, value);
}

class WritebackScheduler {
public:
    WritebackScheduler(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                       const TimelineSink& timeline)
        : schedule(schedule), cost(hardware, dataflow), timeline(timeline) {
        queue_input(schedule.at({0, 0, 0}), nullptr);
    }

    WritebackResult run() {
        for (;;) {
            retire();
            start_array();
            start_dma();
            if (!array && !dma) {
                if (next) throw std::logic_error("writeback scheduler cannot make progress");
                return result;
            }
            advance();
        }
    }

private:
    void queue_input(TileStep step, const TileStep* previous) {
        next = step;
        next_fetch = cost.fetch(step, previous);
        inputs_ready = next_fetch.cycles == 0;
    }

    std::optional<std::size_t> free_bank() const {
        for (std::size_t bank = 0; bank < banks.size(); ++bank)
            if (banks[bank].state == BankState::free) return bank;
        return std::nullopt;
    }

    std::optional<std::size_t> oldest_write() const {
        std::optional<std::size_t> selected;
        for (std::size_t bank = 0; bank < banks.size(); ++bank) {
            if (banks[bank].state == BankState::queued &&
                (!selected || banks[bank].queued_at < banks[*selected].queued_at)) {
                selected = bank;
            }
        }
        return selected;
    }

    void emit(OperationKind operation, EventPhase phase, Shape tile,
              std::optional<std::size_t> bank) const {
        if (timeline) timeline({now, operation, phase, tile, bank});
    }

    void retire() {
        if (array && array->finish == now) {
            emit(OperationKind::array, EventPhase::finish, array->step.index, array->bank);
            if (array->step.last_k) {
                banks[array->bank] = {BankState::queued, array->step, now};
                reduction_bank.reset();
            }
            array.reset();
        }
        if (dma && dma->finish == now) {
            emit(dma->kind, EventPhase::finish, dma->tile, dma->bank);
            if (dma->bank) banks[*dma->bank].state = BankState::free;
            else inputs_ready = true;
            dma.reset();
        }
    }

    void start_array() {
        if (array || !next || !inputs_ready) return;
        const auto step = *next;
        if (step.index.k == 0) {
            reduction_bank = free_bank();
            if (!reduction_bank) return;
            banks[*reduction_bank].state = BankState::reducing;
        }
        if (!reduction_bank) throw std::logic_error("missing accumulator bank for reduction");
        const auto work = cost.array_tile(step.region);
        result.metrics.add(work);
        array = ArrayRun{step, *reduction_bank, checked_add(now, work.cycles.array_time())};
        emit(OperationKind::array, EventPhase::start, step.index, *reduction_bank);

        // At most one read is outstanding. On reuse, the matching operand keeps its
        // current bank while a missed operand can use its other bank.
        if (const auto successor = schedule.next(step.index)) {
            queue_input(schedule.at(*successor), &step);
        } else {
            next.reset();
        }
    }

    void launch_dma(const DmaRequest& request, OperationKind kind, Shape tile,
                    std::optional<std::size_t> bank = {}) {
        request.account(result.metrics);
        auto& service = kind == OperationKind::input_fetch ? result.dma.input_service
                                                          : result.dma.output_service;
        accumulate(service, request.cycles);
        dma = DmaRun{kind, tile, bank, checked_add(now, request.cycles)};
        emit(kind, EventPhase::start, tile, bank);
    }

    void start_dma() {
        if (dma) return;
        // An outstanding input is either needed by an idle array or is its one
        // lookahead. A bank wait with ready inputs falls through to the oldest write.
        if (next && !inputs_ready) {
            launch_dma(next_fetch, OperationKind::input_fetch, next->index);
        } else if (const auto write = oldest_write()) {
            auto& bank = banks[*write];
            bank.state = BankState::writing;
            launch_dma(cost.store(bank.last_step.region), OperationKind::output_write,
                       bank.last_step.index, *write);
        }
    }

    void advance() {
        auto finish = array ? array->finish : dma->finish;
        if (dma) finish = std::min(finish, dma->finish);
        const auto elapsed = finish - now;
        if (!array) {
            accumulate(result.metrics.cycles.hbm_stall, elapsed);
            if (!next) accumulate(result.wait.drain, elapsed);
            else if (!inputs_ready) accumulate(result.wait.input, elapsed);
            else accumulate(result.wait.accumulator, elapsed);
        } else if (dma) {
            auto& hidden = dma->kind == OperationKind::input_fetch ? result.dma.input_hidden
                                                                 : result.dma.output_hidden;
            accumulate(hidden, elapsed);
            accumulate(result.metrics.dma_hidden_cycles, elapsed);
        }
        now = finish;
    }

    const Schedule& schedule;
    TileCost cost;
    const TimelineSink& timeline;
    Count now{};
    WritebackResult result{};
    std::optional<TileStep> next;
    DmaRequest next_fetch{};
    bool inputs_ready{};
    std::optional<ArrayRun> array;
    std::optional<DmaRun> dma;
    std::array<Bank, 2> banks{};
    std::optional<std::size_t> reduction_bank;
};

}  // namespace

void WritebackResult::add(const WritebackResult& other) {
    metrics.add(other.metrics);
    accumulate(dma.input_service, other.dma.input_service);
    accumulate(dma.output_service, other.dma.output_service);
    accumulate(dma.input_hidden, other.dma.input_hidden);
    accumulate(dma.output_hidden, other.dma.output_hidden);
    accumulate(wait.input, other.wait.input);
    accumulate(wait.accumulator, other.wait.accumulator);
    accumulate(wait.drain, other.wait.drain);
}

WritebackResult simulate_writeback(const Hardware& hardware, const Schedule& schedule,
                                   Dataflow dataflow, const TimelineSink& timeline) {
    if (!hardware.double_buffered) throw InputError("writeback requires double buffering");
    return WritebackScheduler(hardware, schedule, dataflow, timeline).run();
}

}  // namespace tensys
