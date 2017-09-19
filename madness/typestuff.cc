#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/MADworld.h>
#include <madness/world/buffer_archive.h>

#include <cstdio>
#include <memory>

using namespace madness;
using namespace std;

namespace c_interface {

    // Have to use a struct here since cannot have partial specialization of functions
    // and below use partial specialization to make templates for arrays.

    // In the default implementation we send nothing in the header, and serialize
    // in one chunk into the payload
    template <typename T>
    struct ugh {
        static int payload_size(const void* t) // t points to an instance of T
        {
            madness::archive::BufferOutputArchive ar;
            ar & (*(T*)t);
            return int(ar.size());
        }

        static int header_size(const void* t) {return 0;}

        static void get_info(const void* t, uint64_t* hs, uint64_t* ps, int* is_contiguous_mask, void** ptr) {
            *hs = 0;
            *ps = payload_size(t);
            *is_contiguous_mask = 0;
            *ptr = nullptr;
        }

        static void pack_header(const void* t, uint64_t header_size, void**ptr) {}

        /// t --- obj to be serialized
        /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
        /// pos --- position in the input buffer to resume serialization
        /// ptr[chunk_size] --- place for output 
        static void pack_payload(const void* t, uint64_t* chunk_size, uint64_t pos, void* ptr)
        {
            madness::archive::BufferOutputArchive ar(ptr, chunk_size);
            ar & (*(T*)t);
        }

        // t points to some memory in which we will construct an object from the header
        static void unpack_header(void* t, uint64_t header_size, const void* header) {
            MADNESS_ASSERT(header_size == 0);
            new(t) T;
        }

        static void unpack_payload(void* t, uint64_t chunk_size, uint64_t pos, void* ptr)
            madness::archive::BufferInputArchive ar(ptr, chunk_size);
            ar & (*(T*)t);
        }
        
        static void print(const void *ptr)
        {
            std::cout << *(T*)ptr << std::endl;
        }
    };

    // This struct and functions within it should be accessible from C
    struct data_descriptor {
        const char* name;
        void (*payload_size)(const void*);
        void (*header_size)(const void*);
        void (*get_info)(const void* t, uint64_t* hs, uint64_t* ps, int* is_contiguous_mask, void** ptr);
        void (*pack_header)(const void* t, uint64_t header_size, void**ptr);
        void (*pack_payload)(const void* t, uint64_t* chunk_size, uint64_t pos, void* ptr);
        void (*unpack_header)(void* t, uint64_t header_size, const void* header);
        void (*unpack_payload)(void* t, uint64_t chunk_size, uint64_t pos, void* ptr);
        void (*print)(const void* t);
    };

    // Returns a pointer to a constant static instance initialized
    // once at run time.  Call this from a piece of C++ code (see
    // example below) and return the pointer to C.
    template <typename T>
    const data_descriptor*
    get_data_descriptor() {
        static const data_descriptor d = {typeid(T).name(),
                                          &ugh<T>::payload_size,
                                          &ugh<T>::header_size,
                                          &ugh<T>::get_info,
                                          &ugh<T>::pack_header,
                                          &ugh<T>::pack_payload,
                                          &ugh<T>::unpack_header,
                                          &ugh<T>::unpack_payload,
                                          &ugh<T>::print};
        return &d;
    }
}

class Fred {
    int value;
public:
    Fred() : value(-1) {}

    Fred(int value) : value(value) {}

    int get() const {return value;}
    
    template <typename Archive>
    void serialize(const Archive& ar) {
        ar & value;
    }
};

std::ostream& operator<<(std::ostream& s, const Fred& f) {
    s << "Fred(" << f.get() << ")";
    return s;
}

// Test code written as if calling from C
template<typename T>
void test(const T& t)
{
    // This line has to be in a piece of C++ that knows the type T
    const c_interface::data_descriptor* d = c_interface::get_data_descriptor<T>();

    // The rest could be in C ... deliberately use printf below rather than C++ streamio
    void* vt = (void*) &t;
    printf("%s %d\n", d->name, d->size(vt));
    
    // Serialize into a buffer
    char buf[256];
    d->serialize(vt, (void *) buf, sizeof(buf));
    
    void* g = d->defconstruct();
    d->deserialize(g, (void*) buf, sizeof(buf));
    printf("deserialize ");
    d->print(g);
    d->destructor((void*) g);

    void *f = d->defconstruct_and_deserialize((void *) buf, sizeof(buf));
    printf("defconstruct and deserialize ");
    d->print(f);
    d->destructor(f);
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    try {

        test(99);
        test(Fred(33));
        test(99.0);
        test(std::vector<Fred>(4,Fred(55)));
        int a[4] = {1,2,3,4};
        test(a); 
        Fred b[4] = {Fred(1),Fred(2),Fred(3),Fred(4)};
        test(b); 
        
    }
    catch (SafeMPI::Exception e) {
        error("caught an MPI exception");
    }
    catch (madness::MadnessException e) {
        print("XXX",e);
        error("caught a MADNESS exception");
    }
    catch (const char* s) {
        print(s);
        error("caught a string exception");
    }
    catch (...) {
        error("caught unhandled exception");
    }

    finalize();
    return 0;
}
