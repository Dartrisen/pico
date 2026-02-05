#pragma once
#include <memory>
#include "IEngine.hpp"


template<class EngineT>
class EngineWrapper final : public IEngine {
public:
    explicit EngineWrapper(EngineT engine)
        : engine_(std::move(engine)) {}

    void advance(double dt) override {
        engine_.advance(dt);
    }

private:
    EngineT engine_;
};
