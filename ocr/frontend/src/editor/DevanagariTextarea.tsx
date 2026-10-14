import { useEffect, useRef, useState } from "react";

const DEV_DIGITS = "०१२३४५६७८९";
const LAT_DIGITS = "0123456789";

function toLatin(s: string): string {
  return Array.from(s).map((c) => {
    const i = DEV_DIGITS.indexOf(c);
    return i >= 0 ? LAT_DIGITS[i] : c;
  }).join("");
}

function toDevanagari(s: string): string {
  return Array.from(s).map((c) => {
    const i = LAT_DIGITS.indexOf(c);
    return i >= 0 ? DEV_DIGITS[i] : c;
  }).join("");
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  onCommit?: (v: string) => void;
  placeholder?: string;
  rows?: number;
  className?: string;
}

export function DevanagariTextarea({
  value, onChange, onCommit, placeholder, rows = 3, className = "",
}: Props) {
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const [lastEdit, setLastEdit] = useState<string>(value);

  useEffect(() => {
    if (value !== lastEdit) setLastEdit(value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  function commit() {
    const nfc = value.normalize("NFC");
    onChange(nfc);
    onCommit?.(nfc);
  }

  return (
    <div className={"space-y-1 " + className}>
      <textarea
        ref={ref}
        value={value}
        rows={rows}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onBlur={commit}
        spellCheck={false}
        className="w-full rounded bg-slate-800 px-2 py-1 font-deva text-sm text-slate-100 outline-none focus:ring-1 focus:ring-sky-500"
      />
      <div className="flex gap-1 text-[10px]">
        <button
          type="button"
          onClick={() => onChange(toDevanagari(value))}
          className="rounded bg-slate-700 px-1.5 py-0.5 text-slate-300 hover:bg-slate-600"
          title="Convert Latin digits to Devanagari"
        >
          0→०
        </button>
        <button
          type="button"
          onClick={() => onChange(toLatin(value))}
          className="rounded bg-slate-700 px-1.5 py-0.5 text-slate-300 hover:bg-slate-600"
          title="Convert Devanagari digits to Latin"
        >
          ०→0
        </button>
        <button
          type="button"
          onClick={commit}
          className="ml-auto rounded bg-slate-700 px-1.5 py-0.5 text-slate-300 hover:bg-slate-600"
        >
          Normalize NFC
        </button>
      </div>
    </div>
  );
}