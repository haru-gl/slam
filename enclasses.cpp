#include "enclasses.h"

matchingType operator*(matchingType ft)
{
    return ft;
}
matchingType operator++(matchingType& ft)
{
    return ft = matchingType(std::underlying_type<matchingType>::type(ft) + 1);

}
std::ostream& operator<<(std::ostream& os, matchingType ft)
{
    switch (ft) {
    case matchingType::mSIMILARITY:
        return os << "mSIMILARITY";
    case matchingType::mAFFINE:
        return os << "mAFFINE";
    case matchingType::mPROJECTIVE:
        return os << "mPROJECTIVE";
    case matchingType::mPROJECTIVE3:
        return os << "mPROJECTIVE3";
    case matchingType::mPROJECTIVE_EV:
        return os << "mPROJECTIVE_EV";
    default: return os;
    }
}
