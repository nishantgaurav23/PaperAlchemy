"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/lib/auth/store";

interface ProtectedRouteProps {
  children: ReactNode;
  fallback?: ReactNode;
}

function DefaultLoading() {
  return (
    <div role="status" className="flex min-h-[200px] items-center justify-center">
      <div className="size-8 animate-spin rounded-full border-4 border-muted border-t-primary" />
    </div>
  );
}

export function ProtectedRoute({ children, fallback }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading } = useAuthStore();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push(`/login?redirect=${encodeURIComponent(pathname)}`);
    }
  }, [isLoading, isAuthenticated, router, pathname]);

  if (isLoading) {
    return <>{fallback ?? <DefaultLoading />}</>;
  }

  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
}
