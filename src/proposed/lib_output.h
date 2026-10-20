
#ifndef LIB_OUTPUT_H
#define LIB_OUTPUT_H

#include <chrono>
#include <format>
#include <iostream>
#include <iomanip>
#include <list>
#include <ranges>
#include <vector>
#include <unordered_set>


/*
  std::pair, std::vector, std::list について、
  std::coutに直接流せるようにoperator<<をオーバーロード
*/
template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T,U>& p){
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec){
  os << "[";
  if(!vec.empty()){
    os << vec.front();
    for(const T &x : vec | std::views::drop(1)){
      os << ", " << x;
    }
  }
  os << "]";
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::list<T>& lis){
  os << "[";
  if(!lis.empty()){
    os << lis.front();
    for(const T &x : lis | std::views::drop(1)){
      os << ", " << x;
    }
  }
  os << "]";
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set){
  os << "{";
  if(!set.empty()){
    auto it = set.begin();
    os << *it;
    it++;
    while(it != set.end()){
      os << ", " << *it;
      it++;
    }
  }
  os << "}";
  return os;
}

std::string getDatetimeStr();

// 時間計測クラス ついでに現在時刻の表示
class Timer{
  public:
  std::chrono::_V2::system_clock::time_point start_time;

  // 現在時刻をstringで返す
  std::string now_str(){
    auto now = std::chrono::system_clock::now();
    auto now_local = std::chrono::zoned_time{"Asia/Tokyo", now};
    // std::format("{:%Y年%m月%d日%H時%M分%S秒}", now_local);
    return std::format("{:%c}", now_local);
  }

  // 計測開始
  void start(){
    this->start_time = std::chrono::system_clock::now();
  }

  // 経過時間表示string
  std::string elapsed_str(){
    auto now = std::chrono::system_clock::now();
    auto elap = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time);
    // std::cout << elap << std::endl; // 123s とか
    return std::format("{:%H:%M:%S}", elap); // 100hを超えるとバグるけど一旦気にしない
  }

  // 経過時間を秒で返す
  int elapsed_sec(){
    auto now = std::chrono::system_clock::now();
    auto elap = std::chrono::duration_cast<std::chrono::seconds>(now - this->start_time);
    return elap.count();
  }
};

void debug_lib_output();

#endif
