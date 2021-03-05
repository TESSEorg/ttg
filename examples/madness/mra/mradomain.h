#ifndef MADDOMAIN_H_INCL
#define MADDOMAIN_H_INCL

#include <array>
#include <iostream>
#include <utility>
#include <cmath>
#include <cassert>

#include "mratypes.h"
#include "mrakey.h"

namespace mra {
    
    template <Dimension NDIM>
    struct Domain {
        // Rather than this static data we might be better off with a singleton object that can be easily serialized/communicated, etc. ?

        // Also, might want to associate functions with a domain to ensure consistency?

        // Eliminated storing both double and float data since the data is accessed outside of inner loops

        // Also, the C++ parser requires Domain<NDIM>:: template get<T>(d) ... ugh!
        
        inline static std::array<std::pair<double,double>,NDIM> cell; // first=lo, second=hi
        inline static std::array<double,NDIM> cell_width;
        inline static std::array<double,NDIM> cell_reciprocal_width;
        inline static double cell_volume;
        inline static bool initialized = false;
        
        static void set(Dimension d, double lo, double hi) {
            assert(d<NDIM);
            assert(hi>lo);
            
            cell[d] = {lo,hi};
            cell_width[d] = hi - lo;
            cell_reciprocal_width[d] = 1.0/cell_width[d];
            cell_volume = 1.0;
            for (double x : cell_width) cell_volume *= x;
            initialized = true;
        }

        static void uninitialize() {initialized=false;}
        
        static void set_cube(double lo, double hi) {
            for (auto d : range(NDIM)) set(d, lo, hi);
        }
        
        /// Returns the simulation domain in dimension d as a pair of values (first=lo, second=hi)
        static const std::pair<double,double>& get(Dimension d) {
            assert(d<NDIM);
            assert(initialized);
            return cell[d];
        }
        
        /// Returns the width of dimension d
        static double get_width(Dimension d) {
            assert(d<NDIM);
            assert(initialized);
            return cell_width[d];
        }

        /// Returns the maximum width of any dimension
        static double get_max_width() {
            double w = 0.0;
            for (auto d : range(NDIM)) w = std::max(w,get_width(d));
            return w;
        }
        
        static double get_reciprocal_width(Dimension d) {
            assert(d<NDIM);
            assert(initialized);
            return cell_reciprocal_width[d];
        }

        template <typename T> 
        static T get_volume() {
            assert(initialized);
            return cell_volume;
        }
        
        /// Convert user coords (Domain) to simulation coords ([0,1]^NDIM)
        template <typename T>
        static void user_to_sim(const Coordinate<T,NDIM>& xuser, Coordinate<T,NDIM>& xsim) {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            for (Dimension d=0; d<NDIM; ++d)
                xsim[d] = (xuser[d] - cell[d].first) * cell_reciprocal_width[d];
            //return xsim;
        }
        
        /// Convert simulation coords ([0,1]^NDIM) to user coords (Domain)
        template <typename T>
        static void sim_to_user(const Coordinate<T,NDIM>& xsim, Coordinate<T,NDIM>& xuser) {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            for (Dimension d=0; d<NDIM; ++d) {
                xuser[d] = xsim[d]*cell_width[d] + cell[d].first;
            }
        }
        
        /// Returns the corners in user coordinates (Domain) that bound the box labelled by the key
        template <typename T>
        static std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>> bounding_box(const Key<NDIM>& key) {
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            Coordinate<T,NDIM> lo, hi;
            const T h = std::pow(T(0.5),T(key.level()));
            const std::array<Translation,NDIM>& l = key.translation();
            for (Dimension d=0; d<NDIM; ++d) {
                T box_width = h*cell_width[d];
                lo[d] = cell[d].first + box_width*l[d];
                hi[d] = lo[d] + box_width;
            }
            return std::make_pair(lo,hi);
        }
        
        /// Returns the box at level n that contains the given point in simulation coordinates
        /// @param[in] pt point in simulation coordinates
        /// @param[in] n the level of the box
        template <typename T>
        static Key<NDIM> sim_to_key(const Coordinate<T,NDIM>& pt, Level n){
            static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, "Domain data only for float or double");
            assert(initialized);
            std::array<Translation,NDIM> l;
            T twon = std::pow(T(2.0), T(n));
            for (Dimension d=0; d<NDIM; ++d) {
                l[d] = Translation(twon*pt[d]);
            }
            return Key<NDIM>(n,l);
        }
        
        static void print() {
            assert(initialized);
            std::cout << "Domain<" << NDIM << ">(" << Domain<NDIM>::cell << ")\n";
        }
    };
}

#endif
