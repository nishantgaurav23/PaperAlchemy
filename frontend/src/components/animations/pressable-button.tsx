"use client";

import { useState, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type PressableButtonProps = ButtonHTMLAttributes<HTMLButtonElement>;

export function PressableButton({
  children,
  className,
  disabled,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  style,
  ...props
}: PressableButtonProps) {
  const [pressed, setPressed] = useState(false);

  return (
    <button
      className={cn(className)}
      disabled={disabled}
      style={{
        transition: "transform 100ms ease",
        transform: pressed && !disabled ? "scale(0.95)" : "scale(1)",
        ...style,
      }}
      onMouseDown={(e) => {
        if (!disabled) setPressed(true);
        onMouseDown?.(e);
      }}
      onMouseUp={(e) => {
        setPressed(false);
        onMouseUp?.(e);
      }}
      onMouseLeave={(e) => {
        setPressed(false);
        onMouseLeave?.(e);
      }}
      {...props}
    >
      {children}
    </button>
  );
}
