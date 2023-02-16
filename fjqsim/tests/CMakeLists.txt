include_directories(../)

add_executable(
        fjqsim_test
        test_statistics.cpp
)

target_link_libraries(
        fjqsim_test
        fjqsimlib
        GTest::gmock
        GTest::gmock_main
)
