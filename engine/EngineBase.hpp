template <class Derived>
class EngineBase
{
public:
    void advance(double dt)
    {
        static_cast<Derived*>(this)->advance_impl(dt);
    }
};
