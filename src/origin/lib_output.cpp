

#include <chrono>
#include <format>
#include <iostream>
#include <iomanip>
#include <list>
#include <ranges>
#include <vector>
#include <unordered_set>
#include "lib_output.h"

#include <mpi.h> // OPEN MPI


// template<typename T, typename U>
// std::ostream& operator<<(std::ostream& os, const std::pair<T,U>& p)

// template<typename T>
// std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)

// template<typename T>
// std::ostream& operator<<(std::ostream& os, const std::list<T>& lis)

// template<typename T>
// std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set)

// webよりコピペ 現在時刻を文字列として取得
std::string getDatetimeStr() {
  time_t t = time(nullptr);
  const tm* localTime = localtime(&t);
  std::stringstream s;
  s << localTime->tm_year + 1900 << "/";
  s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1 << "/";
  s << std::setw(2) << std::setfill('0') << localTime->tm_mday << " ";
  s << std::setw(2) << std::setfill('0') << localTime->tm_hour << ":";
  s << std::setw(2) << std::setfill('0') << localTime->tm_min << ":";
  s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
  return s.str();
}


// ライブラリ内の各コードを実行するテスト
void debug_lib_output(){

  std::pair<int,std::string> x = {123, "abc"};
  std::cout << x << std::endl;


  std::vector<int> vec = {1,2,3,4};
  std::cout << vec << std::endl;

  std::vector<std::vector<int>> vec2d = {{1,2,3},{4},{},{5,6}};
  std::cout << vec2d << std::endl;

  std::list<int> lis = {5,6,7,8};
  std::cout << lis << std::endl;

  std::unordered_set<int> uset = {5,6,7,8};
  std::cout << uset << std::endl;



  std::cout << getDatetimeStr() << std::endl;

  return;
}
