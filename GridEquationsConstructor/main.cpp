// g++ main.cpp -o app
// ./app
#include <iostream>

class DVal // класс "Переменная дифф. уравнения"
{
    public:
        int type = 0;// 0 - неинициализировано, 1 - одномерное, 2 - двумерное, 3 - трехмерное, 
        double hx = 0;
        double hy = 0;
        double hz = 0;
        
        // (i,j,k), (i+1,j,k), (i-1,j,k), (i,j+1,k), (i,j-1,k), (i,j,k+1), (i,j,k-1),
        double koeff[7]{0, 0, 0, 0, 0, 0, 0};// Массив для хранения коэффициентов сеточного уравнения

        double num = 0; // Число, возникающее при аппроксимации г.у. 2 рода
        
        DVal(double hx, double hy, double hz)
        {
            type = 3;
            this->hx = hx;
            this->hy = hy;
            this->hz = hz;
            koeff[0] = 1;
            koeff[1] = 1;
            koeff[2] = 1;
            koeff[3] = 1;
            koeff[4] = 1;
            koeff[5] = 1;
            koeff[6] = 1;
        }
        
        DVal(double hx, double hy)
        {
            type = 2;
            this->hx = hx;
            this->hy = hy;
            koeff[0] = 1;
            koeff[1] = 1;
            koeff[2] = 1;
            koeff[3] = 1;
            koeff[4] = 1;
        }
        
        DVal(double hx)
        {
            type = 1;
            this->hx = hx;
            koeff[0] = 1;
            koeff[1] = 1;
            koeff[2] = 1;
        }

        DVal()
        {
            type = 0;
        }
    
    public:
        /*DVal& operator*(double k)
        {
            koeff[0]*=k;
            koeff[1]*=k;
            koeff[2]*=k;
            koeff[3]*=k;
            koeff[4]*=k;
            koeff[5]*=k;
            koeff[6]*=k;
            return this;
        }*/
        
        void Print()
        {
            if(type == 3)
            {
                std::cout 
                    << "hx = " << hx << ";"
                    << "hy = " << hy << ";"
                    << "hz = " << hz << ";"
                    << "k0: " << koeff[0] << "; "
                    << "k1: " << koeff[1] << "; "
                    << "k2: " << koeff[2] << "; "
                    << "k3: " << koeff[3] << "; "
                    << "k4: " << koeff[4] << "; "
                    << "k5: " << koeff[5] << "; "
                    << "k6: " << koeff[6] << "."
                    << std::endl;
            }
            else if(type == 2)
            {
                std::cout
                    << "hx = " << hx << ";"
                    << "hy = " << hy << ";"
                    << "k0: " << koeff[0] << "; "
                    << "k1: " << koeff[1] << "; "
                    << "k2: " << koeff[2] << "; "
                    << "k3: " << koeff[3] << "; "
                    << "k4: " << koeff[4] << "."
                    << std::endl;
            }
            else if(type == 1)
            {
                std::cout
                    << "hx = " << hx << ";"
                    << "k0: " << koeff[0] << "; "
                    << "k1: " << koeff[1] << "; "
                    << "k2: " << koeff[2] << "."
                    << std::endl;
            }
            else
            {
                std::cout << "Error in type!!!";
                exit(-1);
            }
        }
};

// определяем оператор сложения вне класса
DVal operator + (const DVal& c1, const DVal& c2) 
{
    DVal res;

    int resType = 0;
    if(c1.type > c2.type)
    {
        res.type = c1.type;
        res.hx = c1.hx;
        res.hy = c1.hy;
        res.hz = c1.hz;
    }
    else
    {
        res.type = c2.type;
        res.hx = c2.hx;
        res.hy = c2.hy;
        res.hz = c2.hz;
    }
        

    for(int i = 0; i < 7; i++)
    {
        res.koeff[i] = c1.koeff[i] + c2.koeff[i];
    }
        
    return res;
}

// Оператор второй производной
class DOp
{
    public:
        static DVal& DOp2(DVal& dval)
        {
            double k0 = 0;
            k0 += (-2/(dval.hx*dval.hx));
            if(dval.type>=2)
                k0 += (-2/(dval.hy*dval.hy));
            if(dval.type==3)
                k0 += (-2/(dval.hz*dval.hz));
            
            dval.koeff[0] *= k0;
            dval.koeff[1] *= (1/(dval.hx*dval.hx));
            dval.koeff[2] *= (1/(dval.hx*dval.hx));
            dval.koeff[3] *= (1/(dval.hy*dval.hy));
            dval.koeff[4] *= (1/(dval.hy*dval.hy));
            dval.koeff[5] *= (1/(dval.hz*dval.hz));
            dval.koeff[6] *= (1/(dval.hz*dval.hz));
            return dval;
        }
};

int main()
{
    std::cout << "---1D---" << std::endl;
    DVal T1(2);
    T1.Print();
   (DOp::DOp2(T1)).Print();
    std::cout << "---2D---" << std::endl;
    DVal T2(2,3);
    T2.Print();
   (DOp::DOp2(T2)).Print();
    std::cout << "---3D---" << std::endl;
    DVal T3(2,3,4);
    T3.Print();
   (DOp::DOp2(T3)).Print();
    std::cout << "--------" << std::endl;
    std::cout << "T2: ";
    T2.Print();
    std::cout << "T3: ";
    T3.Print();

    DVal T3pT3 = T3+T3;
    T3pT3.Print();

    return 0;
}