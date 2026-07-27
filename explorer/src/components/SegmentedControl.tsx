interface Option<T extends string | number> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string | number> {
  label: string;
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
}

export function SegmentedControl<T extends string | number>({
  label,
  value,
  options,
  onChange
}: SegmentedControlProps<T>) {
  return (
    <label className="segmented-control">
      <span>{label}</span>
      <span className="segments">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={option.value === value ? "active" : ""}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </span>
    </label>
  );
}
