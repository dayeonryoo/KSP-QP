include(GoogleTest)

# One binary per header under test, mirroring include/. Keeps failures
# localized (a broken solution test can't stop ssn tests from running)
# and keeps individual binaries fast to rebuild while iterating.
#
# The PDE generators are Python (python/pde_generator.py, python/fem_q1.py);
# their tests live in python/tests/ and run under `python3 -m unittest`.
function(ksp_qp_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ksp_qp GTest::gtest_main)
  gtest_discover_tests(${name})
endfunction()

ksp_qp_add_test(test_amd_ordering)
ksp_qp_add_test(test_mps_format_parser)
ksp_qp_add_test(test_problem)
ksp_qp_add_test(test_solution)
ksp_qp_add_test(test_printing)
ksp_qp_add_test(test_schur_operator)
ksp_qp_add_test(test_schur_preconditioner)
ksp_qp_add_test(test_schur_smw_update)
ksp_qp_add_test(test_ssn)
ksp_qp_add_test(test_ksp_qp)
