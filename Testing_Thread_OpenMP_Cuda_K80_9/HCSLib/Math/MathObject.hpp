#pragma once

template<class T, class TProxy = T>
struct MathObject
{
    typedef T       FinalType;
    typedef TProxy  ProxyType;

    FinalType& Self()
    {
        return static_cast<FinalType&>(*this);
    }

    const FinalType& Self() const
    {
        return static_cast<const FinalType&>(*this);
    }

    ProxyType GetProxy() const
    {
        return Self();
    }
};