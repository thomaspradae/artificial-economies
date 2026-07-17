import unittest

from build_publication_inference import benjamini_hochberg, student_t_two_sided_p


class PublicationInferenceTests(unittest.TestCase):
    def test_student_t_two_sided_p_handles_zero_variance(self):
        self.assertEqual(student_t_two_sided_p(0.0, 0.0, 20), 1.0)
        self.assertEqual(student_t_two_sided_p(1.0, 0.0, 20), 0.0)

    def test_benjamini_hochberg_is_monotone_in_sorted_order(self):
        p_values = [0.01, 0.04, 0.03, 0.002]
        adjusted = benjamini_hochberg(p_values)
        ordered = [adjusted[index] for index in sorted(range(len(p_values)), key=p_values.__getitem__)]
        self.assertEqual(ordered, sorted(ordered))
        self.assertAlmostEqual(adjusted[3], 0.008)


if __name__ == "__main__":
    unittest.main()
