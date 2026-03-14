import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { Checkbox } from "./checkbox";

describe("Checkbox", () => {
  it("renders as a checkbox", () => {
    render(<Checkbox aria-label="Accept terms" />);
    expect(screen.getByRole("checkbox", { name: "Accept terms" })).toBeInTheDocument();
  });

  it("toggles checked state on click", async () => {
    const user = userEvent.setup();
    render(<Checkbox aria-label="Toggle" />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    expect(checkbox).toBeChecked();
  });

  it("renders disabled state", () => {
    render(<Checkbox disabled aria-label="Disabled" />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
  });

  it("applies custom className", () => {
    render(<Checkbox className="custom" aria-label="Custom" />);
    expect(screen.getByRole("checkbox")).toHaveClass("custom");
  });
});
