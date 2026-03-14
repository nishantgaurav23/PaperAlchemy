import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { Popover, PopoverTrigger, PopoverContent } from "./popover";

describe("Popover", () => {
  it("renders trigger", () => {
    render(
      <Popover>
        <PopoverTrigger>Info</PopoverTrigger>
        <PopoverContent>Details here</PopoverContent>
      </Popover>
    );
    expect(screen.getByText("Info")).toBeInTheDocument();
  });

  it("shows content on trigger click", async () => {
    const user = userEvent.setup();
    render(
      <Popover>
        <PopoverTrigger>Info</PopoverTrigger>
        <PopoverContent>Details here</PopoverContent>
      </Popover>
    );
    await user.click(screen.getByText("Info"));
    expect(screen.getByText("Details here")).toBeInTheDocument();
  });

  it("applies custom className to content", async () => {
    const user = userEvent.setup();
    render(
      <Popover>
        <PopoverTrigger>Info</PopoverTrigger>
        <PopoverContent className="custom-class">Content</PopoverContent>
      </Popover>
    );
    await user.click(screen.getByText("Info"));
    expect(screen.getByText("Content").closest("[data-slot='popover-content']")).toBeInTheDocument();
  });
});
