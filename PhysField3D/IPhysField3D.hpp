class IPhysField3D
{
public:    
    virtual ~IPhysField3D() {}
    virtual inline size_t GetNx() const = 0;
    virtual inline size_t GetNy() const = 0;
    virtual inline size_t GetNz() const = 0;
    virtual inline void SetValue(size_t i, size_t j, size_t k, double value) = 0;
    virtual inline double GetValue(size_t i, size_t j, size_t k) const = 0;
    virtual void PrintDataArray(size_t indStart, size_t indEnd) const = 0;
    virtual void PrintDataArray() const = 0;
    virtual void PrintValue(size_t i, size_t j, size_t k) const = 0;
};