import { useMemo } from "react";
import { Group, Text } from "react-konva";
import type { BBox } from "../types";

interface Props {
  bbox: BBox;
  text: string;
  color?: string;
  opacity?: number;
  padding?: number;
  minFontSize?: number;
  maxFontSize?: number;
  align?: "left" | "center" | "right";
  verticalAlign?: "top" | "middle" | "bottom";
  bold?: boolean;
  italic?: boolean;
  clip?: boolean;
  /** Starting font size hint — the box still shrinks it if it does not fit. */
  fontSize?: number;
}

/**
 * Renders text fitted to the inside of a bounding box.
 *
 *  - `fontSize` (if given) is used as a HINT, not an override.
 *  - If the wrapped text still does not fit, the font shrinks step-by-step
 *    down to `minFontSize`.
 *  - Overflow is clipped so neighbouring regions never receive stray glyphs.
 */
export function TextInBox({
  bbox,
  text,
  color = "#0f172a",
  opacity = 1,
  padding = 3,
  minFontSize = 6,
  maxFontSize = 28,
  align = "left",
  verticalAlign = "top",
  bold = false,
  italic = false,
  clip = true,
  fontSize: fontSizeHint,
}: Props) {
  const [x0, y0, x1, y1] = bbox;
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const innerW = Math.max(1, w - padding * 2);
  const innerH = Math.max(1, h - padding * 2);

  const fontSize = useMemo(() => {
    if (!text) return minFontSize;

    // start from the hint, otherwise from a fraction of the box height
    let size =
      fontSizeHint != null && fontSizeHint > 0
        ? Math.min(maxFontSize, Math.max(minFontSize, fontSizeHint))
        : Math.min(maxFontSize, Math.max(minFontSize, innerH * 0.7));

    for (let i = 0; i < 32 && size > minFontSize; i++) {
      const lineH = size * 1.2;
      const avgCharW = size * 0.55;
      const charsPerLine = Math.max(1, Math.floor(innerW / avgCharW));
      const lines = Math.ceil(text.length / charsPerLine);
      if (lines * lineH <= innerH) return size;
      size = Math.max(minFontSize, size - 1);
    }
    return minFontSize;
  }, [text, innerW, innerH, minFontSize, maxFontSize, fontSizeHint]);

  if (!text) return null;

  const node = (
    <Text
      x={x0 + padding}
      y={y0 + padding}
      width={innerW}
      height={innerH}
      text={text}
      fontSize={fontSize}
      fontFamily="Noto Sans Devanagari"
      fontStyle={
        bold && italic
          ? "bold italic"
          : bold
            ? "bold"
            : italic
              ? "italic"
              : "normal"
      }
      fill={color}
      opacity={opacity}
      align={align}
      verticalAlign={verticalAlign}
      wrap="word"
      ellipsis
      listening={false}
    />
  );

  if (!clip) return node;
  return (
    <Group
      clipX={x0}
      clipY={y0}
      clipWidth={w}
      clipHeight={h}
      listening={false}
    >
      {node}
    </Group>
  );
}