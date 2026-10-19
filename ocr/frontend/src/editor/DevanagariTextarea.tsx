import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { tokenize, tokenAt, type Token } from "../spell/core";
import { useSpell, isWrong, suggest } from "../spell/useSpell";

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

  const spellOn = useSpell((s) => s.enabled);
  // re-render when results arrive / words are ignored or added
  useSpell((s) => s.version);
  useSpell((s) => s.ignored);
  useSpell((s) => s.userWords);

  const tokens = useMemo(() => (spellOn ? tokenize(value) : []), [spellOn, value]);
  const wrong = spellOn ? tokens.filter((t) => isWrong(t.word) === true) : [];

  const [caret, setCaret] = useState<number | null>(null);
  const caretWord = caret == null ? undefined : tokenAt(tokens, caret);
  const current = caretWord && isWrong(caretWord.word) === true ? caretWord : undefined;

  // the underline layer must wrap exactly like the textarea (minus its scrollbar)
  const backdrop = useRef<HTMLDivElement | null>(null);
  const [innerWidth, setInnerWidth] = useState<number | undefined>(undefined);
  useLayoutEffect(() => {
    const ta = ref.current;
    if (!ta || !spellOn) return;
    const sync = () => {
      setInnerWidth(ta.clientWidth);
      if (backdrop.current) backdrop.current.scrollTop = ta.scrollTop;
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(ta);
    return () => ro.disconnect();
  }, [spellOn, value]);

  function replace(t: Token, by: string) {
    const next = value.slice(0, t.start) + by + value.slice(t.end);
    onChange(next);
    const pos = t.start + by.length;
    requestAnimationFrame(() => {
      ref.current?.focus();
      ref.current?.setSelectionRange(pos, pos);
      setCaret(pos);
    });
  }

  function jumpTo(t: Token) {
    ref.current?.focus();
    ref.current?.setSelectionRange(t.start, t.end);
    setCaret(t.start);
  }

  const taClass =
    "w-full rounded px-2 py-1 font-deva text-sm text-slate-100 outline-none focus:ring-1 focus:ring-sky-500 " +
    (spellOn ? "relative bg-transparent" : "bg-slate-800");

  return (
    <div className={"space-y-1 " + className}>
      <div className={spellOn ? "relative rounded bg-slate-800" : ""}>
        {spellOn && (
          <div
            ref={backdrop}
            aria-hidden
            className="pointer-events-none absolute left-0 top-0 h-full overflow-hidden whitespace-pre-wrap break-words px-2 py-1 font-deva text-sm text-transparent"
            style={{ width: innerWidth }}
          >
            <Underlined text={value} wrong={wrong} />
          </div>
        )}
        <textarea
          ref={ref}
          value={value}
          rows={rows}
          placeholder={placeholder}
          onChange={(e) => { onChange(e.target.value); setCaret(e.target.selectionStart); }}
          onSelect={(e) => setCaret((e.target as HTMLTextAreaElement).selectionStart)}
          onScroll={(e) => { if (backdrop.current) backdrop.current.scrollTop = (e.target as HTMLTextAreaElement).scrollTop; }}
          onBlur={commit}
          spellCheck={false}
          className={taClass}
        />
      </div>
      {spellOn && current && (
        <SuggestBar
          key={current.word + current.start}
          token={current}
          onPick={(s) => replace(current, s)}
        />
      )}
      {spellOn && !current && wrong.length > 0 && (
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => jumpTo(wrong[0])}
          className="text-[10px] text-rose-300 hover:underline"
          title="Jump to the first underlined word"
        >
          {wrong.length} possible misspelling{wrong.length > 1 ? "s" : ""} — click an underlined word
        </button>
      )}
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
/** The text with misspelled words wavy-underlined (text itself invisible). */
function Underlined({ text, wrong }: { text: string; wrong: Token[] }) {
  const parts: ReactNode[] = [];
  let at = 0;
  wrong.forEach((t, i) => {
    parts.push(<Fragment key={"t" + i}>{text.slice(at, t.start)}</Fragment>);
    parts.push(
      <span
        key={"w" + i}
        style={{
          textDecorationLine: "underline",
          textDecorationStyle: "wavy",
          textDecorationColor: "#f43f5e",
          textDecorationSkipInk: "none",
          textUnderlineOffset: "3px",
        }}
      >
        {text.slice(t.start, t.end)}
      </span>,
    );
    at = t.end;
  });
  parts.push(<Fragment key="end">{text.slice(at)}</Fragment>);
  // a trailing newline needs something after it to take up its line
  if (text.endsWith("\n")) parts.push(<Fragment key="nl">{"\u200b"}</Fragment>);
  return <>{parts}</>;
}

/** Suggestions + Ignore / Add to dictionary for the misspelled word at the cursor. */
function SuggestBar({ token, onPick }: { token: Token; onPick: (s: string) => void }) {
  const ignore = useSpell((s) => s.ignore);
  const addWord = useSpell((s) => s.addWord);
  const [list, setList] = useState<string[] | null>(null);
  useEffect(() => {
    let live = true;
    void suggest(token.word).then((s) => { if (live) setList(s); });
    return () => { live = false; };
  }, [token.word]);

  // buttons must not take focus from the textarea
  const keep = (e: ReactMouseEvent) => e.preventDefault();
  const btn = "rounded px-1.5 py-0.5 text-[10px]";
  return (
    <div className="flex flex-wrap items-center gap-1 rounded border border-rose-900/60 bg-slate-900 p-1">
      <span className="font-deva text-[11px] text-rose-300">{token.word}</span>
      <span className="text-[10px] text-slate-500">→</span>
      {list === null && <span className="text-[10px] text-slate-500">…</span>}
      {list?.length === 0 && <span className="text-[10px] text-slate-500">no suggestions</span>}
      {list?.map((s) => (
        <button key={s} type="button" onMouseDown={keep} onClick={() => onPick(s)}
          className={btn + " bg-slate-700 font-deva text-[11px] text-slate-100 hover:bg-sky-700"}>
          {s}
        </button>
      ))}
      <span className="ml-auto flex gap-1">
        <button type="button" onMouseDown={keep} onClick={() => ignore(token.word)}
          className={btn + " bg-slate-800 text-slate-300 hover:bg-slate-700"}
          title="Stop underlining this word until the browser is closed">
          Ignore
        </button>
        <button type="button" onMouseDown={keep} onClick={() => addWord(token.word)}
          className={btn + " bg-slate-800 text-slate-300 hover:bg-slate-700"}
          title="Never underline this word again (saved in this browser)">
          Add to dictionary
        </button>
      </span>
    </div>
  );
}
