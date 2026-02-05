#pragma once

struct IEngine {
    virtual ~IEngine() = default;
    virtual void advance(double dt) = 0;
};
