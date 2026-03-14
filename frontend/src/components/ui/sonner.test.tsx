import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Toaster } from "./sonner";

describe("Toaster (Sonner)", () => {
  it("renders without crashing", () => {
    const { container } = render(<Toaster />);
    expect(container).toBeInTheDocument();
  });

  it("renders toaster section element", () => {
    const { container } = render(<Toaster />);
    const section = container.querySelector("section");
    expect(section).toBeInTheDocument();
  });
});
