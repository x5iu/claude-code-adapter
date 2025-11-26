package profile

import "context"

// contextKey is a private type used for context keys to avoid collisions.
type contextKey struct{}

// profileKey is the context key for storing the Profile.
var profileKey = contextKey{}

// WithProfile returns a new context with the given profile attached.
func WithProfile(ctx context.Context, p *Profile) context.Context {
	return context.WithValue(ctx, profileKey, p)
}

// FromContext retrieves the Profile from the context.
// Returns nil and false if no profile is set.
func FromContext(ctx context.Context) (*Profile, bool) {
	p, ok := ctx.Value(profileKey).(*Profile)
	return p, ok
}

// MustFromContext retrieves the Profile from the context.
// Panics if no profile is set.
func MustFromContext(ctx context.Context) *Profile {
	p, ok := FromContext(ctx)
	if !ok {
		panic("profile: no profile in context")
	}
	return p
}
