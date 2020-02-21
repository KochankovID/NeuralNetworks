#pragma once

template<typename T, typename Y>
class Func {
public:
	Func() {};
	virtual Y operator()(const T&) = 0;
	virtual ~Func() {};
};