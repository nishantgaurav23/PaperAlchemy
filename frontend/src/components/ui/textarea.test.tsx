import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { Textarea } from "./textarea";

describe("Textarea", () => {
  it("renders with placeholder", () => {
    render(<Textarea placeholder="Type here..." />);
    expect(screen.getByPlaceholderText("Type here...")).toBeInTheDocument();
  });

  it("renders with data-slot attribute", () => {
    render(<Textarea placeholder="test" />);
    expect(screen.getByPlaceholderText("test")).toHaveAttribute("data-slot", "textarea");
  });

  it("handles disabled state", () => {
    render(<Textarea disabled placeholder="disabled" />);
    expect(screen.getByPlaceholderText("disabled")).toBeDisabled();
  });

  it("accepts user input", async () => {
    const user = userEvent.setup();
    render(<Textarea placeholder="type" />);
    const textarea = screen.getByPlaceholderText("type");
    await user.type(textarea, "Hello world");
    expect(textarea).toHaveValue("Hello world");
  });

  it("applies custom className", () => {
    render(<Textarea className="my-class" placeholder="test" />);
    expect(screen.getByPlaceholderText("test")).toHaveClass("my-class");
  });
});
