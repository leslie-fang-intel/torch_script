#include <iostream>
using namespace std;

inline unsigned char max_propagate_nan(unsigned char a, unsigned char b) {
  return a > b ? a : b;
}
extern "C" void kernel(const unsigned char* in_ptr0,
                       const unsigned char* in_ptr1,
                       const unsigned char* in_ptr2,
                       unsigned char* out_ptr0,
                       long* out_ptr1)
{
    {
        {
            long tmp_acc0 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33024L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(0L)];
                auto tmp4 = in_ptr2[static_cast<long>(0L)];
                auto tmp2 = max_propagate_nan(tmp0, tmp1);
                auto tmp5 = max_propagate_nan(tmp2, tmp4);
                auto tmp6 = decltype(tmp5)(tmp5 * tmp5);
                auto tmp7 = static_cast<long>(tmp6);
                out_ptr0[static_cast<long>(x0)] = tmp1;
                tmp_acc0 = tmp_acc0 + tmp7;
            }
            out_ptr1[static_cast<long>(0L)] = tmp_acc0;
        }
    }
}

int main(int argc,char ** argv){
	unsigned char in_ptr0[33024];
    unsigned char in_ptr1[1];
    unsigned char in_ptr2[1];
    for ( int i = 0; i < 33024; i++ ) {
       in_ptr0[i] = 6;
    }
    for ( int i = 0; i < 1; i++ ) {
       in_ptr1[i] = 6;
       in_ptr2[i] = 6;
    }
    unsigned char out_ptr0[33024];
    long out_ptr1[1];
    kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1);
    std::cout<<"out_ptr1 is: "<<out_ptr1[0]<<std::endl;
	return 1;
}

