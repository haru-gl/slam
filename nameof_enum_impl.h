#pragma once

/*
@file nameof_enum_impl.h
@brief nameof_enum.hの実装部分。見づらいのでファイル分割してある
@author inaenomaki
*/


// 定義に基づいて適切なマクロを選択（`__FUNCSIG__`がMVCC専用であり，clangおよびgccでコンパイルに失敗するため）
#if defined(_MSC_VER)
    #define FUNC_SIGNATURE __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
    #define FUNC_SIGNATURE __PRETTY_FUNCTION__
#else
    #define FUNC_SIGNATURE "Unknown compiler"
#endif


#include<cstdint>

namespace nameof_enum {
	namespace _not_for_user {
		namespace {

			/**
			@namespace _not_for_user
			@brief 内部的な処理。テンプレートなので仕方なくここに書いているが、基本的に外から使用されることを想定していない
			*/

			//! ENUMを検索する範囲のデフォルト値。テンプレートの特殊化をして展開する際に使用する
			constexpr int32_t ENUM_SEARCH_MIN = -256;
			constexpr int32_t ENUM_SEARCH_MAX = 256;

			/**
			@brief シグネチャを返す。テンプレート引数として非型パラメータを渡すことで、シグネチャにenumの具体的な値を取得する足がかりにする
			@tparam E 型。enumかenum class。
			@tparam V 値。これが何かを取得したい。
			*/
			template<typename E, E V>
			const char* get_signature() {
				static_assert(std::is_enum<E>::value == true, "enumもしくはenum class以外の名前のnameofには対応していません");
				return FUNC_SIGNATURE;//事前定義マクロはコンパイラ依存
			}

			/**
			@brief get_signatureを走査するためにintで呼べるするようにする為のラッパ
			@tparam E 型。enumかenum class。
			@tparam V 値。数値。Eにキャストされる。
			*/
			template<typename E, int32_t V>
			const char* get_signature_int() {
				return get_signature<E, static_cast<E>(V)>();
			}

			/**
			@brief 関数テンプレートで再帰ループをする為のラッパ
			@tparam I ループに使う非型テンプレート引数
			@tparam MAX ループの最大
			@note 関数テンプレートの部分特殊化が出来ないので、ラッパクラスを用意した
			*/
			template<typename E, int32_t I, int32_t MAX>
			struct SearchSignatureLoop {
				static_assert(I <= MAX, "Maxよりも大きい値をsearchしようとしています。範囲を確認してください。");

				//! 検索単位。1度の再帰で探索するenumの数。
				static constexpr int32_t ENUM_SEARCH_UNIT = 32;

				/**
				@brief ループ関数。与えられた値と一致するenumを探し、それをテンプレート引数にとる関数のシグネチャを返す。
				@param[in] value 名前を取得したいenumの値
				*/
				static const char* search(E value) {
					switch ((int)value) {		// ここをint型にしないと, enum class型で失敗する．
					case I:
						return get_signature_int<E, I>();
					case I + 1:
						return get_signature_int<E, I + 1>();
					case I + 2:
						return get_signature_int<E, I + 2>();
					case I + 3:
						return get_signature_int<E, I + 3>();
					case I + 4:
						return get_signature_int<E, I + 4>();
					case I + 5:
						return get_signature_int<E, I + 5>();
					case I + 6:
						return get_signature_int<E, I + 6>();
					case I + 7:
						return get_signature_int<E, I + 7>();
					case I + 8:
						return get_signature_int<E, I + 8>();
					case I + 9:
						return get_signature_int<E, I + 9>();
					case I + 10:
						return get_signature_int<E, I + 10>();
					case I + 11:
						return get_signature_int<E, I + 11>();
					case I + 12:
						return get_signature_int<E, I + 12>();
					case I + 13:
						return get_signature_int<E, I + 13>();
					case I + 14:
						return get_signature_int<E, I + 14>();
					case I + 15:
						return get_signature_int<E, I + 15>();
					case I + 16:
						return get_signature_int<E, I + 16>();
					case I + 17:
						return get_signature_int<E, I + 17>();
					case I + 18:
						return get_signature_int<E, I + 18>();
					case I + 19:
						return get_signature_int<E, I + 19>();
					case I + 20:
						return get_signature_int<E, I + 20>();
					case I + 21:
						return get_signature_int<E, I + 21>();
					case I + 22:
						return get_signature_int<E, I + 22>();
					case I + 23:
						return get_signature_int<E, I + 23>();
					case I + 24:
						return get_signature_int<E, I + 24>();
					case I + 25:
						return get_signature_int<E, I + 25>();
					case I + 26:
						return get_signature_int<E, I + 26>();
					case I + 27:
						return get_signature_int<E, I + 27>();
					case I + 28:
						return get_signature_int<E, I + 28>();
					case I + 29:
						return get_signature_int<E, I + 29>();
					case I + 30:
						return get_signature_int<E, I + 30>();
					case I + 31:
						return get_signature_int<E, I + 31>();
					default:
						break;
					}

					constexpr int32_t NEXT_BASE = I + ENUM_SEARCH_UNIT;
					constexpr int32_t NEXT_I = NEXT_BASE > MAX ? MAX : NEXT_BASE;
					return SearchSignatureLoop<E, NEXT_I, MAX>::search(value);//終了
				}
			};

			/**
			@brief 関数テンプレートで再帰ループをする為のラッパ
			@note 最大値とIが等しいとき。こいつでループを止める。
			*/
			template<typename E, int32_t I>
			struct SearchSignatureLoop <E, I, I> {
				static const char* search(E value) {
					return "not found";
				}
			};

			/**
			@brief シグネチャを走査して取得する
			*/
			template<typename EnumType>
			const char* search_signature(EnumType value) {
				return _not_for_user::SearchSignatureLoop<EnumType, _not_for_user::ENUM_SEARCH_MIN, _not_for_user::ENUM_SEARCH_MAX>::search(value);
			}

			/**
			@brief シグネチャを走査して取得する
			@tparam MIN 走査範囲の最小値
			@tparam MAX 走査範囲の最大値
			@note [MIN,MAX]の範囲外のものについてはenumの名前を見つけることができない
			*/
			template<typename EnumType, int32_t MIN, int32_t MAX>
			const char* search_signature(EnumType value) {
				return _not_for_user::SearchSignatureLoop<EnumType, MIN, MAX>::search(value);
			}

			/*
			@brief シグネチャから、enumの名前の部分を取り出す。
			@param[in] signature シグネチャ
			@param[in] omitsNamespace 名前空間の表記を省略するか
			@note VSでは以下の形で出る
			const char *__cdecl nameof_enum::_not_for_user::`anonymous-namespace'::get_signature<enum Enum,X>(void)
			上記のフォーマットなのを前提としたゴリ押しのVS固有処理なので注意
			*/
			std::string extract_nameof(const char* signature, bool omitsNamespace) {
				constexpr char FUNC_NAME[] = "get_signature";
				constexpr size_t FUNC_NAME_SIZE = sizeof(FUNC_NAME);
				const std::string sig(signature);
				const size_t funcPos = sig.find(FUNC_NAME);//関数名の開始位置
				const size_t namePartPos = sig.find(",", funcPos) + 1;//enumの値の名前部分の開始位置
				const size_t nameEndPos = sig.find(">", funcPos) - 1;//enumの値の名前部分の終了位置

				//名前空間を省略する時は最後の":"を探す
				size_t extractStartPos = namePartPos;
				if (omitsNamespace) {
					extractStartPos = sig.rfind(":", nameEndPos) + 1;
					if (extractStartPos == std::string::npos || extractStartPos < namePartPos) {
						extractStartPos = namePartPos;
					}
				}

				const size_t length = nameEndPos - extractStartPos + 1;

				if (funcPos != std::string::npos &&
					nameEndPos != std::string::npos &&
					extractStartPos != std::string::npos &&
					length > 0)
				{
					return sig.substr(extractStartPos, length);
				}
				else {
					return "parse error";
				}
			}

		}// end of namespace `anonymous`
	}// end of namespace _not_for_user
}// end of namespace of nameof_enum