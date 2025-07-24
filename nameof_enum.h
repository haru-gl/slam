/*
@file nameof_enum.h
@brief enum(enum class)の名前取得関数を提供
@author inaenomaki
*/

#pragma once
#include<string>
#include"nameof_enum_impl.h"

namespace nameof_enum {

	/**
	@brief enumの名前を取得する
	@param[in] value 名前を取得したいEnumの値
	*/
	template<typename EnumType>
	std::string nameof(EnumType value, bool omitsNamespace = true) {
		return _not_for_user::extract_nameof(_not_for_user::search_signature(value), omitsNamespace);
	}

	/**
	@brief enumの名前を取得する
	@param[in] value 名前を取得したいEnumの値
	@tparam MIN Enumの最小値
	@tparam MAX Enumの最大値
	*/
	template<typename EnumType, int32_t MIN, int32_t MAX>
	std::string nameof(EnumType value, bool omitsNamespace = true) {
		return _not_for_user::extract_nameof(_not_for_user::search_signature<EnumType, MIN, MAX>(value), omitsNamespace);
	}

}// end of namespace of nameof_enum
