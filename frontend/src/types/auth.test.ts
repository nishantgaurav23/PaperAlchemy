import { describe, it, expect } from "vitest";
import {
  UserSchema,
  LoginResponseSchema,
  SignupResponseSchema,
  LoginRequestSchema,
  SignupRequestSchema,
  ForgotPasswordRequestSchema,
} from "./auth";

describe("Auth Zod Schemas", () => {
  const validUser = {
    id: "user-1",
    email: "test@example.com",
    name: "Test User",
    created_at: "2025-01-01T00:00:00Z",
  };

  describe("UserSchema", () => {
    it("validates a valid user", () => {
      const result = UserSchema.safeParse(validUser);
      expect(result.success).toBe(true);
    });

    it("validates user with optional fields", () => {
      const result = UserSchema.safeParse({
        ...validUser,
        avatar_url: "https://example.com/avatar.png",
        affiliation: "MIT",
      });
      expect(result.success).toBe(true);
    });

    it("rejects user missing required fields", () => {
      const result = UserSchema.safeParse({ id: "user-1" });
      expect(result.success).toBe(false);
    });

    it("rejects user with invalid email", () => {
      const result = UserSchema.safeParse({ ...validUser, email: "not-an-email" });
      expect(result.success).toBe(false);
    });
  });

  describe("LoginRequestSchema", () => {
    it("validates valid login request", () => {
      const result = LoginRequestSchema.safeParse({
        email: "test@example.com",
        password: "password123",
      });
      expect(result.success).toBe(true);
    });

    it("rejects login with short password", () => {
      const result = LoginRequestSchema.safeParse({
        email: "test@example.com",
        password: "short",
      });
      expect(result.success).toBe(false);
    });

    it("rejects login with invalid email", () => {
      const result = LoginRequestSchema.safeParse({
        email: "bad",
        password: "password123",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("SignupRequestSchema", () => {
    it("validates valid signup request", () => {
      const result = SignupRequestSchema.safeParse({
        email: "test@example.com",
        password: "password1",
        name: "Test User",
      });
      expect(result.success).toBe(true);
    });

    it("validates signup with optional affiliation", () => {
      const result = SignupRequestSchema.safeParse({
        email: "test@example.com",
        password: "password1",
        name: "Test User",
        affiliation: "MIT",
      });
      expect(result.success).toBe(true);
    });

    it("rejects signup with password missing number", () => {
      const result = SignupRequestSchema.safeParse({
        email: "test@example.com",
        password: "password",
        name: "Test User",
      });
      expect(result.success).toBe(false);
    });

    it("rejects signup without name", () => {
      const result = SignupRequestSchema.safeParse({
        email: "test@example.com",
        password: "password1",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("LoginResponseSchema", () => {
    it("validates valid login response", () => {
      const result = LoginResponseSchema.safeParse({
        access_token: "jwt-token-here",
        token_type: "bearer",
        user: validUser,
      });
      expect(result.success).toBe(true);
    });

    it("rejects response missing token", () => {
      const result = LoginResponseSchema.safeParse({
        token_type: "bearer",
        user: validUser,
      });
      expect(result.success).toBe(false);
    });

    it("rejects response with invalid user", () => {
      const result = LoginResponseSchema.safeParse({
        access_token: "jwt-token-here",
        token_type: "bearer",
        user: { id: "bad" },
      });
      expect(result.success).toBe(false);
    });
  });

  describe("SignupResponseSchema", () => {
    it("validates valid signup response (same shape as login)", () => {
      const result = SignupResponseSchema.safeParse({
        access_token: "jwt-token-here",
        token_type: "bearer",
        user: validUser,
      });
      expect(result.success).toBe(true);
    });
  });

  describe("ForgotPasswordRequestSchema", () => {
    it("validates valid email", () => {
      const result = ForgotPasswordRequestSchema.safeParse({
        email: "test@example.com",
      });
      expect(result.success).toBe(true);
    });

    it("rejects invalid email", () => {
      const result = ForgotPasswordRequestSchema.safeParse({
        email: "not-valid",
      });
      expect(result.success).toBe(false);
    });
  });
});
